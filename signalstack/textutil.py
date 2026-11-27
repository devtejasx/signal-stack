"""Text helpers: HTML extraction, keyword mining, URL canonicalisation, dates, similarity."""
from __future__ import annotations

import hashlib
import html as htmllib
import re
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP = set(ENGLISH_STOP_WORDS) | {
    "com", "www", "http", "https", "click", "read", "learn", "contact", "privacy", "policy", "terms", "cookie",
    "cookies", "login", "sign", "signup", "menu", "home", "page", "search", "copyright", "rights", "reserved",
    "follow", "subscribe", "newsletter", "email", "phone", "address", "submit", "request", "free", "get", "us",
    "best", "top", "new", "now", "today", "know", "view", "explore", "help", "need", "want", "make", "use",
    "just", "like", "also", "one", "two", "may", "will", "can", "said", "says", "year", "years", "time", "day",
    "including", "according", "news", "latest", "update", "updates", "per", "via", "within", "across", "around",
    "nbsp", "amp", "quot", "rs", "inr", "lakh", "crore", "pm", "am", "ist",
}

COUNTRIES = [
    "India", "USA", "United States", "US", "UK", "United Kingdom", "Canada", "Australia", "Germany", "France",
    "Ireland", "New Zealand", "Singapore", "Dubai", "UAE", "Netherlands", "Italy", "Spain", "Japan", "China",
    "Sweden", "Switzerland", "Denmark", "Finland", "Norway", "Poland", "Malaysia", "Hong Kong", "South Korea",
    "Indonesia", "Vietnam", "Bangladesh", "Nepal", "Sri Lanka", "Pakistan", "Nigeria", "Kenya", "Brazil",
    "Mexico", "Russia", "Europe", "Asia", "Africa", "Middle East", "Saudi Arabia", "Qatar", "Philippines",
]

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid",
                   "ref", "ref_src", "mc_cid", "mc_eid", "ocid", "cvid", "ei", "form"}


def clean_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(s or "")).strip()


def html_to_text(html: str, max_chars: int = 60000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "iframe", "form", "template"]):
        t.decompose()
    for t in soup.find_all(["nav", "footer"]):
        t.decompose()
    text = soup.get_text("\n")
    lines = [clean_ws(l) for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 2]
    out, seen = [], set()
    for l in lines:
        if l in seen:
            continue
        seen.add(l)
        out.append(l)
    return "\n".join(out)[:max_chars]


def strip_tags(s: str | None) -> str:
    if not s:
        return ""
    if "<" not in s:
        return clean_ws(s)
    return clean_ws(BeautifulSoup(s, "lxml").get_text(" "))


def page_meta(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    def meta(name: str) -> str:
        t = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        return clean_ws(t.get("content")) if t and t.get("content") else ""

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"].strip())
        if href.startswith(("http://", "https://")):
            links.append((clean_ws(a.get_text(" "))[:120], href.split("#")[0]))
    feeds = []
    for l in soup.find_all("link", attrs={"rel": True, "href": True}):
        rel = " ".join(l.get("rel") or []).lower()
        typ = (l.get("type") or "").lower()
        if "alternate" in rel and ("rss" in typ or "atom" in typ):
            feeds.append(urljoin(base_url, l["href"]))
    headings = [clean_ws(h.get_text(" ")) for h in soup.find_all(["h1", "h2", "h3"])]
    headings = [h for h in headings if 3 < len(h) < 160]
    return {
        "title": clean_ws(soup.title.get_text()) if soup.title else "",
        "description": meta("description") or meta("og:description"),
        "keywords": meta("keywords"),
        "og_site_name": meta("og:site_name"),
        "headings": list(dict.fromkeys(headings))[:60],
        "links": links,
        "feeds": list(dict.fromkeys(feeds)),
    }


def canonical_url(url: str) -> str:
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() not in TRACKING_PARAMS]
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunparse(("https", netloc, path, "", urlencode(q), ""))


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def domain_of(url: str) -> str:
    try:
        d = urlparse(url if "://" in url else "https://" + url).netloc.lower()
    except ValueError:
        return ""
    return d[4:] if d.startswith("www.") else d


def normalise_website(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


def parse_date(v) -> str | None:
    """Return ISO-8601 UTC string or None."""
    if not v:
        return None
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, (tuple, list)) and len(v) >= 6:  # time.struct_time from feedparser
        dt = datetime(*v[:6], tzinfo=timezone.utc)
    else:
        s = str(v).strip()
        dt = None
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError, IndexError):
            pass
        if dt is None:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%S",
                        "%Y%m%dT%H%M%SZ", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)


_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "") if w.lower() not in STOP and len(w) > 2]


def key_phrases(texts: list[str], top: int = 25) -> list[str]:
    """Frequent uni/bi-grams (cheap, deterministic keyword mining)."""
    c: Counter = Counter()
    for t in texts:
        toks = tokens(t)
        c.update(toks)
        c.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    scored = []
    for k, n in c.items():
        if n < 2:
            continue
        weight = n * (1.8 if " " in k else 1.0)
        scored.append((weight, k))
    scored.sort(reverse=True)
    out: list[str] = []
    for _, k in scored:
        if any(k in o or o in k for o in out):
            continue
        out.append(k)
        if len(out) >= top:
            break
    return out


def find_countries(text: str) -> list[str]:
    found = []
    for c in COUNTRIES:
        if re.search(rf"(?<![A-Za-z]){re.escape(c)}(?![A-Za-z])", text or ""):
            found.append(c)
    norm = {"United States": "USA", "US": "USA", "United Kingdom": "UK"}
    return list(dict.fromkeys(norm.get(c, c) for c in found))


def similarity_matrix(docs: list[str]):
    vec = TfidfVectorizer(stop_words=list(STOP), ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=40000)
    X = vec.fit_transform([d or "" for d in docs])
    return cosine_similarity(X)


def sim_to_reference(reference: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    m = similarity_matrix([reference, *docs])
    return [float(x) for x in m[0, 1:]]


def truncate(s: str | None, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"
