"""Initial company research.

Reads the company's own website (robots-permitted pages only), its sitemap, Wikipedia,
recent news mentions and app-store presence, then builds the first company profile.

Facts vs assumptions
--------------------
* A **fact** is a statement whose words could be matched against the text of a page we
  actually fetched (the page URL is stored with it).
* An **assumption** is an inference (by the LLM or by heuristics) that could not be matched
  to fetched text. Assumptions are shown separately and are confirmed or corrected by the
  founder in the interview.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from . import adapters, db, llm
from .fetch import fetcher
from .jobs import JobContext
from .textutil import (clean_ws, domain_of, find_countries, html_to_text, key_phrases, normalise_website,
                       page_meta, tokens, truncate)

KEY_PAGE_PATTERNS = [
    ("about", r"about|who-we-are|our-story|company|team"),
    ("services", r"services|solutions|products?|programs?|offerings?|features|what-we-do|platform|courses"),
    ("pricing", r"pricing|plans|fees|packages|premium"),
    ("customers", r"testimonials?|reviews?|success|case-stud|customers|stories"),
    ("news", r"blog|news|press|media|insights|articles|updates"),
    ("faq", r"faq|help|support"),
    ("careers", r"careers?|jobs|join-us|hiring"),
    ("contact", r"contact"),
]

PROFILE_SYSTEM = """You are a meticulous business analyst building an intelligence profile of a company
from text scraped from its own website plus a few public sources. Separate what the text says
(facts, each tied to the URL it came from) from what you infer (assumptions). Never invent numbers,
customers, funding or people that are not in the text. Be specific to THIS company."""

PROFILE_PROMPT = """Company: {name}
Website: {website}

=== Evidence (each block starts with its URL) ===
{evidence}

Return JSON with exactly these keys:
{{
 "summary": "2-3 sentence description of what the company does, for whom, where",
 "industry": "specific industry / sub-sector",
 "business_model": "how it most likely makes money (B2C, B2B, marketplace, commission, subscription ...)",
 "offerings": ["specific products/services named in the evidence"],
 "customer_segments": ["specific customer groups"],
 "geographies": ["countries/regions it serves or that its offering concerns"],
 "value_props": ["claims the company makes about why customers should choose it"],
 "revenue_streams": ["likely revenue streams (these are assumptions unless stated)"],
 "category_keywords": ["6-10 short search phrases describing the problem space, e.g. 'study abroad counselling'"],
 "customer_problem_queries": ["5-7 search phrases customers would use when they have the problem this company solves (no company names)"],
 "market_topics": ["6-8 market/industry/technology developments that would materially affect this business"],
 "regulatory_topics": ["3-6 specific regulations, policies, regulators or government programs that affect this business"],
 "competitor_candidates": [{{"name": "...", "domain": "official domain like example.com", "kind": "direct|substitute|adjacent", "why": "one sentence"}}],
 "publications": [{{"name": "trade publication, regulator or industry body", "domain": "domain", "why": "why it is a credible, specific source"}}],
 "facts": [{{"statement": "short factual statement copied closely from the evidence", "source_url": "URL of the evidence block it came from"}}],
 "assumptions": [{{"statement": "an inference", "basis": "what it is inferred from"}}]
}}
Give 8-12 competitor_candidates (real companies you are confident exist, with their real domains),
6-10 publications, 8-15 facts and 4-8 assumptions."""


def _pick_key_pages(links: list[tuple[str, str]], domain: str) -> dict[str, str]:
    picked: dict[str, str] = {}
    for kind, pat in KEY_PAGE_PATTERNS:
        rx = re.compile(pat, re.I)
        for text, href in links:
            if domain_of(href) != domain:
                continue
            path = urlparse(href).path
            if path.count("/") > 3 or len(path) < 2:
                continue
            if rx.search(path) or rx.search(text or ""):
                picked.setdefault(kind, href)
                break
    return picked


def _sitemap_summary(website: str) -> dict:
    f = fetcher()
    rp = f._robots_for(website)  # noqa: SLF001 - reuse cached robots parser
    maps = []
    try:
        maps = list(rp.site_maps() or []) if rp else []
    except Exception:  # noqa: BLE001
        maps = []
    if not maps:
        maps = [website.rstrip("/") + "/sitemap.xml"]
    urls: list[str] = []
    for sm in maps[:2]:
        r = f.get(sm)
        if not r.ok:
            continue
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
        if locs and all(l.endswith(".xml") for l in locs[:5]):  # sitemap index
            for child in locs[:4]:
                cr = f.get(child)
                if cr.ok:
                    urls += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", cr.text)
        else:
            urls += locs
    sections: dict[str, int] = {}
    for u in urls:
        seg = urlparse(u).path.strip("/").split("/")[0] or "(root)"
        sections[seg] = sections.get(seg, 0) + 1
    top = sorted(sections.items(), key=lambda kv: -kv[1])[:12]
    return {"sitemaps": maps[:2], "url_count": len(urls), "sections": top}


def _verify_fact(statement: str, page_text: str) -> float:
    st = set(tokens(statement))
    if not st:
        return 0.0
    pt = set(tokens(page_text))
    return len(st & pt) / len(st)


def research_company(company_id: int, ctx: JobContext | None = None) -> dict:
    step = ctx.step if ctx else (lambda m: None)
    c = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    website, domain, name = c["website"], c["domain"], c["name"]
    f = fetcher()
    db.execute("DELETE FROM research_pages WHERE company_id=?", (company_id,))
    db.execute("DELETE FROM facts WHERE company_id=?", (company_id,))

    step(f"Fetching homepage {website} (robots.txt checked)")
    home = f.get(website)
    pages: list[dict] = []
    meta = {"title": "", "description": "", "keywords": "", "headings": [], "links": [], "feeds": []}
    if home.ok:
        meta = page_meta(home.text, home.final_url or website)
        text = html_to_text(home.text)
        pages.append({"url": home.final_url or website, "kind": "homepage", "title": meta["title"], "text": text,
                      "status": home.status})
    else:
        step(f"Homepage not readable: {home.error or home.status}")

    key = _pick_key_pages(meta["links"], domain)
    step(f"Reading key pages: {', '.join(key) or 'none found'}")
    for kind, url in list(key.items())[:8]:
        r = f.get(url)
        if r.ok:
            m = page_meta(r.text, url)
            pages.append({"url": url, "kind": kind, "title": m["title"], "text": html_to_text(r.text, 15000),
                          "status": r.status})
            meta["feeds"] += m["feeds"]

    step("Reading sitemap to understand site structure")
    sitemap = _sitemap_summary(website)

    step("Looking up Wikipedia")
    wiki = adapters.wikipedia_lookup(name, domain)
    if wiki:
        pages.append({"url": wiki["url"], "kind": "wikipedia", "title": wiki["title"], "text": wiki["extract"], "status": 200})

    step("Collecting recent public news mentions")
    news_src = {"query": f'"{name}"', "type": "bing_news"}
    news = adapters.bing_news(news_src)
    mentions = []
    name_l = name.lower()
    for rec in news.records:
        blob = (rec["title"] + " " + rec["snippet"]).lower()
        if name_l in blob or domain.split(".")[0] in blob.replace(" ", ""):
            mentions.append(rec)
    if mentions:
        pages.append({"url": adapters.bing_news_url(f'"{name}"'), "kind": "news",
                      "title": f"News mentions of {name}",
                      "text": "\n".join(f"{m['title']} — {m['snippet']} ({m['publisher']}, {m['published_at']})" for m in mentions[:15]),
                      "status": 200})

    step("Checking App Store presence")
    apps = []
    for cc in ("in", "us"):
        for a in adapters.itunes_search(name, cc, 6):
            seller_url = (a.get("sellerUrl") or "").lower()
            seller = (a.get("sellerName") or "").lower() + " " + (a.get("artistName") or "").lower()
            if domain in seller_url or name_l.replace(" ", "") in seller.replace(" ", "") or \
                    name_l.replace(" ", "") in (a.get("trackName", "").lower().replace(" ", "")):
                apps.append({"store": "apple", "app_id": a["trackId"], "name": a.get("trackName"), "country": cc,
                             "url": a.get("trackViewUrl"), "rating": a.get("averageUserRating"),
                             "rating_count": a.get("userRatingCount"), "seller": a.get("sellerName")})
        if apps:
            break

    social = {}
    for text, href in meta["links"]:
        d = domain_of(href)
        for net in ("linkedin.com", "twitter.com", "x.com", "instagram.com", "facebook.com", "youtube.com"):
            if d.endswith(net):
                social.setdefault(net, href)
        if "play.google.com" in d:
            social.setdefault("google_play", href)
        if "apps.apple.com" in d:
            social.setdefault("app_store", href)

    for p in pages:
        db.insert("research_pages", {"company_id": company_id, "url": p["url"], "kind": p["kind"], "title": p["title"],
                                     "text": p["text"], "http_status": p["status"], "fetched_at": db.now()})

    # ---------------------------------------------------------------- synthesis
    corpus = "\n\n".join(p["text"] for p in pages)
    countries = find_countries(corpus)
    name_toks = set(re.findall(r"[a-z]+", (name + " " + domain.split(".")[0]).lower()))
    meta_kw = [k.strip().lower() for k in (meta["keywords"] or "").split(",") if 2 <= len(k.split()) <= 4]
    raw = key_phrases([p["text"] for p in pages if p["kind"] != "news"] + [meta["description"]], 40)
    # Prefer multi-word phrases and drop anything that is just the company's own name.
    phrases = [k for k in meta_kw + sorted(raw, key=lambda k: -len(k.split()))
               if not set(k.split()) <= name_toks and not (set(k.split()) & name_toks and len(k.split()) == 1)]
    phrases = list(dict.fromkeys(phrases))[:30]
    evidence = "\n\n".join(f"[{p['url']}] ({p['kind']})\n{truncate(p['text'], 2600 if p['kind'] == 'homepage' else 1500)}"
                           for p in pages)
    step(f"Synthesising profile with {llm.active_provider_label()}")
    data, used = llm.chat_json(PROFILE_SYSTEM, PROFILE_PROMPT.format(name=name, website=website, evidence=evidence[:16000]),
                               purpose="research_profile", max_tokens=4000)
    profile = _rule_profile(name, website, domain, meta, pages, phrases, countries)
    if isinstance(data, dict):
        for k in ("summary", "industry", "business_model"):
            if data.get(k):
                profile[k] = clean_ws(str(data[k]))
        for k in ("offerings", "customer_segments", "geographies", "value_props", "revenue_streams",
                  "category_keywords", "customer_problem_queries", "market_topics", "regulatory_topics"):
            if isinstance(data.get(k), list) and data[k]:
                profile[k] = [clean_ws(str(x)) for x in data[k] if str(x).strip()][:15]
        if isinstance(data.get("competitor_candidates"), list):
            profile["competitor_candidates"] = [x for x in data["competitor_candidates"] if isinstance(x, dict) and x.get("name")]
        if isinstance(data.get("publications"), list):
            profile["publications"] = [x for x in data["publications"] if isinstance(x, dict) and x.get("domain")]
    profile["research_by"] = used
    profile["apps"] = apps
    profile["social"] = social
    profile["feeds"] = list(dict.fromkeys(meta["feeds"]))[:5]
    profile["sitemap"] = sitemap
    profile["wikipedia"] = {"title": wiki["title"], "url": wiki["url"]} if wiki else None
    profile["news_mentions"] = [{"title": m["title"], "url": m["url"], "publisher": m["publisher"],
                                 "published_at": m["published_at"]} for m in mentions[:10]]
    profile["key_pages"] = key

    # ---------------------------------------------------------------- facts / assumptions
    page_by_url = {p["url"]: p for p in pages}
    fact_rows: list[dict] = []
    if meta["title"]:
        fact_rows.append(("fact", "positioning", f"Homepage title: “{meta['title']}”", pages[0]["url"], "verbatim from page <title>"))
    if meta["description"]:
        fact_rows.append(("fact", "positioning", f"Self-description: “{truncate(meta['description'], 300)}”", pages[0]["url"],
                          "verbatim from meta description"))
    for h in meta["headings"][:6]:
        fact_rows.append(("fact", "offering", f"Homepage heading: “{h}”", pages[0]["url"] if pages else website, "verbatim heading"))
    if sitemap["url_count"]:
        secs = ", ".join(f"/{s} ({n})" for s, n in sitemap["sections"][:6])
        fact_rows.append(("fact", "site_structure", f"Sitemap lists {sitemap['url_count']} URLs; largest sections: {secs}",
                          sitemap["sitemaps"][0], "counted from sitemap.xml"))
    if wiki:
        fact_rows.append(("fact", "background", truncate(wiki["extract"], 400), wiki["url"],
                          "Wikipedia" + (" (official domain linked from article)" if wiki["domain_confirmed"] else "")))
    for a in apps[:2]:
        fact_rows.append(("fact", "channels", f"iOS app “{a['name']}” by {a['seller']}: rating {a['rating']} from "
                                              f"{a['rating_count']} ratings ({a['country'].upper()} store)", a["url"], "Apple iTunes Search API"))
    for m in mentions[:3]:
        fact_rows.append(("fact", "news", f"News: {m['title']} ({m['publisher']})", m["url"], "public news index (Bing News RSS)"))
    if isinstance(data, dict):
        for fct in data.get("facts", [])[:15]:
            if not isinstance(fct, dict) or not fct.get("statement"):
                continue
            src = fct.get("source_url") or ""
            page = page_by_url.get(src)
            score = _verify_fact(fct["statement"], page["text"]) if page else max(
                (_verify_fact(fct["statement"], p["text"]) for p in pages), default=0)
            if not page:
                best = max(pages, key=lambda p: _verify_fact(fct["statement"], p["text"]), default=None)
                src = best["url"] if best else src
            if score >= 0.6:
                fact_rows.append(("fact", "llm_extracted", fct["statement"], src,
                                  f"matched {int(score * 100)}% of its words in the cited page"))
            else:
                fact_rows.append(("assumption", "llm_extracted", fct["statement"], src,
                                  f"LLM-stated fact only matched {int(score * 100)}% in fetched text - treated as an assumption"))
        for a in data.get("assumptions", [])[:10]:
            if isinstance(a, dict) and a.get("statement"):
                fact_rows.append(("assumption", "inference", a["statement"], None, a.get("basis") or "model inference"))
    else:
        fact_rows.append(("assumption", "industry", f"Industry inferred from site vocabulary: {', '.join(phrases[:6])}", None,
                          "keyword frequency on fetched pages (no LLM available)"))
    seen_st: list[str] = []
    for kind, field, st, url, ver in fact_rows:
        from rapidfuzz import fuzz
        if any(fuzz.token_set_ratio(st, s) >= 90 for s in seen_st):
            continue  # small models repeat themselves
        seen_st.append(st)
        db.insert("facts", {"company_id": company_id, "kind": kind, "field": field, "statement": st,
                            "source_url": url, "verification": ver, "created_at": db.now()})

    from .profile import save_profile  # local import to avoid a cycle
    save_profile(company_id, profile, "initial research")
    db.update("companies", company_id, {"status": "researched", "updated_at": db.now()})
    step(f"Research complete: {len(pages)} pages, {len(fact_rows)} facts/assumptions, {len(apps)} apps, {len(mentions)} news mentions")
    return profile


def _rule_profile(name, website, domain, meta, pages, phrases, countries) -> dict:
    """Deterministic profile used when no LLM is reachable (and as the base the LLM refines)."""
    desc = meta.get("description") or ""
    headings = meta.get("headings", [])
    return {
        "name": name, "website": website, "domain": domain,
        "summary": desc or (headings[0] if headings else f"{name} ({domain})"),
        "industry": ", ".join(phrases[:3]),
        "business_model": "unknown - confirm in interview",
        "offerings": [h for h in headings if 3 < len(h.split()) < 12][:8],
        "customer_segments": [],
        "geographies": countries[:10],
        "value_props": [h for h in headings if any(w in h.lower() for w in ("best", "trusted", "#1", "expert", "free", "personal"))][:5],
        "revenue_streams": [],
        "category_keywords": phrases[:10],
        "customer_problem_queries": [f"{p} problems" for p in phrases[:4]],
        "market_topics": [f"{p} market" for p in phrases[:4]],
        "regulatory_topics": [f"{p} regulation" for p in phrases[:2]],
        "competitor_candidates": [],
        "publications": [],
        "intelligence": {},
    }
