"""Source adapters. Each adapter turns one approved source into normalised records.

Normalised record: {uid, title, url, snippet, published_at, publisher, extra}

Access policy
-------------
* Web pages and feeds: robots.txt is checked for every request (fetch.Fetcher).
* Official public APIs (Wikipedia Action API, Stack Exchange API, HN Algolia API,
  GDELT DOC API, Apple iTunes Search/RSS) are used under their published API
  policies and rate limits (see fetch.HOST_DELAYS).
* Sources whose terms forbid anonymous crawling (e.g. Reddit) are only used through
  their official API when the user supplies free credentials; otherwise the source is
  recorded as "needs_credentials" and is never scraped.
"""
from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import feedparser

from . import config, db
from .fetch import FetchResult, fetcher
from .textutil import (canonical_url, clean_ws, domain_of, html_to_text, page_meta, parse_date, sha,
                       strip_tags, truncate)


class AdapterResult:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.status = "ok"          # ok | not_modified | error | robots_blocked | needs_credentials | challenge
        self.http_status = 0
        self.error = ""
        self.latency_ms = 0
        self.etag = None
        self.last_modified = None

    def from_fetch(self, r: FetchResult) -> "AdapterResult":
        self.http_status, self.latency_ms = r.status, r.latency_ms
        if r.blocked == "robots":
            self.status, self.error = "robots_blocked", r.error
        elif r.blocked:
            self.status, self.error = "error" if r.blocked != "challenge" else "challenge", r.error
        elif r.not_modified:
            self.status = "not_modified"
        elif r.error:
            self.status, self.error = "error", r.error
        self.etag = r.headers.get("etag")
        self.last_modified = r.headers.get("last-modified")
        return self


# --------------------------------------------------------------------------- feeds
def parse_feed(text: str, base_url: str = "") -> list[dict]:
    feed = feedparser.parse(text)
    out = []
    for e in feed.entries[:60]:
        link = e.get("link") or ""
        pub = parse_date(e.get("published_parsed") or e.get("updated_parsed") or e.get("published") or e.get("updated"))
        summary = strip_tags(e.get("summary") or e.get("description") or "")
        publisher = ""
        if e.get("source") and isinstance(e.get("source"), dict):
            publisher = e["source"].get("title", "")
        publisher = publisher or e.get("news_source") or feed.feed.get("title", "") or domain_of(base_url)
        out.append({
            "uid": e.get("id") or link or e.get("title", ""),
            "title": clean_ws(e.get("title", ""))[:400],
            "url": link,
            "snippet": truncate(summary, 700),
            "published_at": pub,
            "publisher": clean_ws(publisher)[:120],
        })
    return out


def rss(source: dict) -> AdapterResult:
    res = AdapterResult()
    r = fetcher().get(source["url"], etag=source.get("etag"), last_modified=source.get("last_modified"),
                      accept="application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5")
    res.from_fetch(r)
    if res.status != "ok":
        return res
    res.records = parse_feed(r.text, source["url"])
    if not res.records and "<html" in r.text[:500].lower():
        res.status, res.error = "error", "URL did not return a feed"
    return res


def _unwrap_bing(link: str) -> str:
    if "bing.com" in link and "url=" in link:
        qs = parse_qs(urlparse(link).query)
        if qs.get("url"):
            return qs["url"][0]
    return link


def bing_news_url(query: str) -> str:
    return f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss&setlang=en"


def bing_news(source: dict) -> AdapterResult:
    """Bing News public RSS feed (robots.txt permits /news/)."""
    res = AdapterResult()
    url = source.get("url") or bing_news_url(source["query"])
    r = fetcher().get(url, accept="application/rss+xml")
    res.from_fetch(r)
    if res.status != "ok":
        return res
    recs = parse_feed(r.text, url)
    for rec in recs:
        rec["url"] = _unwrap_bing(rec["url"])
        rec["uid"] = canonical_url(rec["url"])
        if rec["publisher"].endswith("BingNews") or rec["publisher"].startswith('"'):
            rec["publisher"] = domain_of(rec["url"])
    feed = feedparser.parse(r.text)
    for rec, e in zip(recs, feed.entries):
        src = e.get("news_source")
        if src:
            rec["publisher"] = clean_ws(src)
    res.records = recs
    return res


def gdelt(source: dict) -> AdapterResult:
    """GDELT DOC 2.0 API - global news index, free, max 1 request / 5 s."""
    res = AdapterResult()
    params = db.loads(source.get("params"), {}) or {}
    q = source["query"]
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + quote_plus(q + " sourcelang:english") +
           f"&mode=artlist&format=json&maxrecords={params.get('max', 50)}&timespan={params.get('timespan', '30d')}&sort=datedesc")
    r = fetcher().get(url, check_robots=False)
    res.from_fetch(r)
    if res.status != "ok":
        return res
    try:
        data = json.loads(r.text or "{}")
    except ValueError:
        res.status, res.error = "error", truncate(r.text, 200) or "invalid JSON"
        return res
    for a in data.get("articles", []):
        res.records.append({
            "uid": canonical_url(a.get("url", "")),
            "title": clean_ws(a.get("title", "")),
            "url": a.get("url", ""),
            "snippet": "",
            "published_at": parse_date(a.get("seendate")),
            "publisher": a.get("domain", ""),
        })
    return res


def hn(source: dict) -> AdapterResult:
    res = AdapterResult()
    url = f"https://hn.algolia.com/api/v1/search_by_date?query={quote_plus(source['query'])}&tags=(story,comment)&hitsPerPage=40"
    r = fetcher().get(url, check_robots=False)
    res.from_fetch(r)
    if res.status != "ok":
        return res
    data = json.loads(r.text or "{}")
    for h in data.get("hits", []):
        title = h.get("title") or h.get("story_title") or ""
        text = strip_tags(h.get("comment_text") or h.get("story_text") or "")
        link = f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        res.records.append({
            "uid": link,
            "title": clean_ws(("Comment on: " if h.get("comment_text") else "") + title)[:300],
            "url": link,
            "snippet": truncate(text, 700),
            "published_at": parse_date(h.get("created_at")),
            "publisher": "Hacker News",
            "extra": {"points": h.get("points"), "external_url": h.get("url")},
        })
    return res


def stackexchange(source: dict) -> AdapterResult:
    res = AdapterResult()
    params = db.loads(source.get("params"), {}) or {}
    site = params.get("site", "stackoverflow")
    url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=creation&pagesize=30"
           f"&q={quote_plus(source['query'])}&site={site}&filter=withbody")
    r = fetcher().get(url, check_robots=False)
    res.from_fetch(r)
    if res.status != "ok":
        return res
    data = json.loads(r.text or "{}")
    for it in data.get("items", []):
        res.records.append({
            "uid": it.get("link"),
            "title": clean_ws(strip_tags(it.get("title", ""))),
            "url": it.get("link"),
            "snippet": truncate(strip_tags(it.get("body", "")), 700),
            "published_at": parse_date(datetime.fromtimestamp(it.get("creation_date", 0), tz=timezone.utc)),
            "publisher": f"{site}.stackexchange.com" if "." not in site else site,
            "extra": {"tags": it.get("tags"), "score": it.get("score"), "answers": it.get("answer_count")},
        })
    return res


def appstore_reviews(source: dict) -> AdapterResult:
    """Apple's public customer-reviews RSS feed (real customer language)."""
    res = AdapterResult()
    params = db.loads(source.get("params"), {}) or {}
    app_id, cc = params.get("app_id"), params.get("country", "in")
    url = f"https://itunes.apple.com/{cc}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    r = fetcher().get(url)
    res.from_fetch(r)
    if res.status != "ok":
        return res
    try:
        entries = json.loads(r.text).get("feed", {}).get("entry", [])
    except ValueError:
        res.status, res.error = "error", "invalid JSON"
        return res
    if isinstance(entries, dict):
        entries = [entries]
    app_url = params.get("app_url") or f"https://apps.apple.com/{cc}/app/id{app_id}"
    for e in entries:
        if "im:rating" not in e:
            continue
        rid = e.get("id", {}).get("label", "")
        rating = e.get("im:rating", {}).get("label")
        res.records.append({
            "uid": f"appstore:{app_id}:{rid}",
            "title": f"★{rating} — {clean_ws(e.get('title', {}).get('label', ''))}",
            "url": f"{app_url}?see-all=reviews#review-{rid}",
            "snippet": truncate(clean_ws(e.get("content", {}).get("label", "")), 700),
            "published_at": parse_date(e.get("updated", {}).get("label")),
            "publisher": f"App Store review ({params.get('app_name', app_id)})",
            "extra": {"rating": int(rating) if rating and rating.isdigit() else None},
        })
    return res


def _last_snapshot(source_id: int) -> dict | None:
    return db.one("SELECT * FROM page_snapshots WHERE source_id=? ORDER BY id DESC LIMIT 1", (source_id,))


def webpage(source: dict) -> AdapterResult:
    """Change detection on a specific page (pricing, product, admissions requirements...)."""
    res = AdapterResult()
    r = fetcher().get(source["url"])
    res.from_fetch(r)
    if res.status != "ok":
        return res
    text = html_to_text(r.text, 40000)
    h = sha(re.sub(r"\d{1,2}:\d{2}|\b\d{4}-\d{2}-\d{2}\b", "", text))  # ignore clocks/dates that change every load
    prev = _last_snapshot(source["id"])
    if prev and prev["content_hash"] == h:
        return res
    db.insert("page_snapshots", {"source_id": source["id"], "fetched_at": db.now(), "content_hash": h, "text": text})
    if not prev:
        return res  # first visit = baseline, not a development
    old_lines = set(prev["text"].splitlines())
    added = [l for l in text.splitlines() if l not in old_lines and len(l) > 25]
    removed = [l for l in prev["text"].splitlines() if l not in set(text.splitlines()) and len(l) > 25]
    if not added and not removed:
        return res
    ratio = difflib.SequenceMatcher(None, prev["text"][:20000], text[:20000]).ratio()
    title = page_meta(r.text, source["url"])["title"] or source["name"]
    snippet = ("Added: " + " | ".join(added[:5]) if added else "") + (" Removed: " + " | ".join(removed[:3]) if removed else "")
    res.records.append({
        "uid": f"change:{source['id']}:{h}",
        "title": f"Page changed: {title}",
        "url": source["url"] + f"#change-{h[:8]}",
        "snippet": truncate(snippet, 900),
        "published_at": db.now(),
        "publisher": domain_of(source["url"]),
        "extra": {"similarity_to_previous": round(ratio, 3), "added_lines": len(added), "removed_lines": len(removed)},
    })
    return res


_ARTICLE_PATH = re.compile(r"/(blog|news|press|insights|articles?|stories|updates|media|resources|posts?)/[^/?#]{6,}", re.I)


def site_updates(source: dict) -> AdapterResult:
    """New article links on a listing page (blog/news/press) of a site without an RSS feed."""
    res = AdapterResult()
    r = fetcher().get(source["url"])
    res.from_fetch(r)
    if res.status != "ok":
        return res
    meta = page_meta(r.text, source["url"])
    dom = domain_of(source["url"])
    links = []
    for text, href in meta["links"]:
        if domain_of(href) != dom or len(text) < 25:
            continue
        if not _ARTICLE_PATH.search(urlparse(href).path) and urlparse(href).path.count("/") < 2:
            continue
        links.append((text, href))
    links = list(dict.fromkeys(links))[:60]
    current = "\n".join(sorted(h for _, h in links))
    h = sha(current)
    prev = _last_snapshot(source["id"])
    if prev and prev["content_hash"] == h:
        return res
    db.insert("page_snapshots", {"source_id": source["id"], "fetched_at": db.now(), "content_hash": h, "text": current})
    known = set(prev["text"].splitlines()) if prev else set()
    for text, href in links:
        if href in known:
            continue
        res.records.append({
            "uid": canonical_url(href), "title": clean_ws(text)[:300], "url": href, "snippet": "",
            "published_at": None if not prev else db.now(), "publisher": dom,
            "extra": {"baseline": not prev},
        })
    return res


def reddit(source: dict) -> AdapterResult:
    res = AdapterResult()
    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        res.status = "needs_credentials"
        res.error = ("Reddit's robots.txt and Public Content Policy forbid anonymous crawling. Add free official API "
                     "credentials (REDDIT_CLIENT_ID/SECRET) to enable this source.")
        return res
    import httpx
    tok = httpx.post("https://www.reddit.com/api/v1/access_token", auth=(config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET),
                     data={"grant_type": "client_credentials"}, headers={"User-Agent": config.USER_AGENT}, timeout=20)
    tok.raise_for_status()
    params = db.loads(source.get("params"), {}) or {}
    sub = params.get("subreddit")
    q = source.get("query") or ""
    api = f"https://oauth.reddit.com/r/{sub}/search?q={quote_plus(q)}&restrict_sr=1&sort=new&limit=50" if q else \
        f"https://oauth.reddit.com/r/{sub}/new?limit=50"
    r = httpx.get(api, headers={"User-Agent": config.USER_AGENT, "Authorization": f"bearer {tok.json()['access_token']}"}, timeout=20)
    res.http_status = r.status_code
    if r.status_code != 200:
        res.status, res.error = "error", f"HTTP {r.status_code}"
        return res
    for c in r.json().get("data", {}).get("children", []):
        d = c["data"]
        link = "https://www.reddit.com" + d.get("permalink", "")
        res.records.append({"uid": link, "title": d.get("title", ""), "url": link,
                            "snippet": truncate(d.get("selftext", ""), 700),
                            "published_at": parse_date(datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)),
                            "publisher": f"r/{sub}"})
    return res


ADAPTERS = {
    "rss": rss,
    "bing_news": bing_news,
    "gdelt": gdelt,
    "hn": hn,
    "stackexchange": stackexchange,
    "appstore_reviews": appstore_reviews,
    "webpage": webpage,
    "site_updates": site_updates,
    "reddit": reddit,
}


def run_adapter(source: dict) -> AdapterResult:
    fn = ADAPTERS.get(source["type"])
    if fn is None:
        res = AdapterResult()
        res.status, res.error = "error", f"No adapter for type {source['type']}"
        return res
    return fn(source)


# --------------------------------------------------------------------------- discovery helpers
def discover_feed(url: str) -> str | None:
    """Find an RSS/Atom feed for a site: <link rel=alternate>, then common paths."""
    f = fetcher()
    r = f.get(url)
    if r.ok:
        if r.text.lstrip()[:200].lower().startswith(("<?xml", "<rss", "<feed")) and parse_feed(r.text):
            return url
        meta = page_meta(r.text, r.final_url or url)
        for feed_url in meta["feeds"]:
            if "comments" in feed_url:
                continue
            fr = f.get(feed_url)
            if fr.ok and parse_feed(fr.text):
                return feed_url
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    for path in ("/feed/", "/rss", "/feed", "/rss.xml", "/atom.xml", "/blog/feed/", "/news/feed/"):
        fr = f.get(urljoin(base, path))
        if fr.ok and fr.text.lstrip()[:300].lower().startswith(("<?xml", "<rss", "<feed")) and parse_feed(fr.text):
            return urljoin(base, path)
    return None


def itunes_search(term: str, country: str = "in", limit: int = 5) -> list[dict]:
    url = (f"https://itunes.apple.com/search?term={quote_plus(term)}&entity=software&country={country}&limit={limit}")
    r = fetcher().get(url)
    if not r.ok:
        return []
    try:
        return json.loads(r.text).get("results", [])
    except ValueError:
        return []


def wikipedia_lookup(name: str, domain: str) -> dict | None:
    f = fetcher()
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=5&srsearch="
           + quote_plus(f'"{name}"'))
    r = f.get(url, check_robots=False)
    if not r.ok:
        return None
    try:
        hits = json.loads(r.text).get("query", {}).get("search", [])
    except ValueError:
        return None
    name_l = name.lower().replace(" ", "")
    for h in hits:
        title = h["title"]
        if name_l not in title.lower().replace(" ", ""):
            continue
        er = f.get("https://en.wikipedia.org/w/api.php?action=query&prop=extracts|extlinks&exintro=1&explaintext=1"
                   f"&ellimit=50&format=json&titles={quote_plus(title)}", check_robots=False)
        if not er.ok:
            continue
        pages = json.loads(er.text).get("query", {}).get("pages", {})
        for p in pages.values():
            links = [l.get("*", "") for l in p.get("extlinks", [])]
            extract = p.get("extract", "")
            if any(domain in l for l in links) or name.lower() in extract.lower()[:300]:
                return {"title": title, "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                        "extract": extract[:3000], "domain_confirmed": any(domain in l for l in links)}
    return None
