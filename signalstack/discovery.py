"""Signal Stack builder: intelligent, justified source discovery.

Pipeline
--------
1. Competitors: candidates come from the research LLM, the founder's interview answers and
   competitor mentions in category news. Every candidate is *verified* by fetching its real
   homepage and measuring TF-IDF similarity against the company's own site + profile. Nothing
   is hard-coded; unreachable or dissimilar candidates are rejected with the reason stored.
2. Candidate sources are generated per category (customers / competitors / market / internal)
   from the profile, then each one is *probed* (fetched once) to measure:
      specific  - share/similarity of its recent records to the company's interest document
      recent    - age of the newest record
      credible  - source-type / publisher reliability prior
   and a human-readable reason is written from that evidence.
3. Sources scoring above AUTO_APPROVE_THRESHOLD are approved; the rest wait in the Source
   Manager for manual approval. Access-restricted sources are listed as 'blocked' with why.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus, urlparse

from . import adapters, config, db, llm
from .fetch import fetcher
from .jobs import JobContext
from .profile import get_profile, interest_document, profile_summary_for_prompt
from .textutil import (COUNTRIES, age_days, clean_ws, domain_of, html_to_text, page_meta, sim_to_reference, tokens,
                       truncate)

TYPE_CREDIBILITY = {
    "webpage": 0.9,        # primary source for changes on that company's own site
    "site_updates": 0.85,
    "rss": 0.75,           # refined per publisher below
    "bing_news": 0.7,
    "gdelt": 0.65,
    "appstore_reviews": 0.6,  # authentic customer voice; individual claims unverified
    "stackexchange": 0.55,
    "hn": 0.5,
    "reddit": 0.45,
    "upload": 0.8,
}

GOV_SUFFIXES = (".gov", ".gov.uk", ".gc.ca", ".gov.au", ".gov.in", ".nic.in", ".europa.eu", ".gov.sg", ".govt.nz", ".edu")

# Catalogue of Stack Exchange communities. Which ones are used is decided per company by
# measuring hits for that company's own problem queries (data-driven, not hard-coded per company).
SE_SITES = ["stackoverflow", "superuser", "serverfault", "softwareengineering", "academia", "expatriates", "travel",
            "money", "workplace", "ux", "webapps", "law", "dba", "datascience", "security", "pm", "startups",
            "freelancing", "cooking", "fitness", "parenting", "health"]


def credibility_for(stype: str, url: str = "") -> float:
    d = "." + domain_of(url or "")
    base = TYPE_CREDIBILITY.get(stype, 0.5)
    if d.endswith(GOV_SUFFIXES):
        return 0.95
    return base


# --------------------------------------------------------------------------- competitors
COMP_NEWS_SYSTEM = """You identify competitors from news headlines. Only list organisations that appear in the
headlines/snippets AND plausibly offer products/services similar to (or substitutes for) the company.
Never invent organisations that are not in the text."""

COMP_NEWS_PROMPT = """Company: {profile}

Headlines and snippets from category news:
{headlines}

Return JSON: {{"competitors": [{{"name": "...", "domain": "their official domain if you know it, else empty", "why": "...", "evidence_url": "URL of the headline"}}]}}"""


DOMAIN_SYSTEM = """You map business names to their official websites. Only give a domain you are confident is the
organisation's real official site. If a name is a generic category of substitute (e.g. "local agents", "neighbourhood
butchers") rather than one organisation, say so."""

DOMAIN_PROMPT = """Context - the company these are competitors of: {profile}

Names:
{names}

Return JSON: {{"names": [{{"name": "<as given>", "is_organisation": true, "domain": "example.com or empty", "search_query": "short news query for a generic category, else empty"}}]}}"""

JUDGE_SYSTEM = """You verify competitor candidates for a market-intelligence system. You are given the company's profile and,
for each candidate, text actually fetched from the candidate's live homepage. Judge ONLY from that homepage evidence:
- "direct": sells a similar offering to the same customers
- "substitute": a different way the same customers solve the same problem
- "adjacent": same industry but a different customer or offering (partner, supplier, marketplace, media)
- "unrelated": different business, parked/for-sale domain, directory, or the page gives no evidence
Be strict: generic words in common (e.g. "students", "fresh", "invest") are not enough."""

JUDGE_PROMPT = """Company: {profile}

Candidates (homepage evidence):
{cands}

Return JSON: {{"verdicts": [{{"key": "<key>", "verdict": "direct|substitute|adjacent|unrelated", "why": "one specific sentence citing the homepage evidence"}}]}}"""


def _llm_domains(company_id: int, p: dict, names: list[str]) -> dict[str, dict]:
    if not names:
        return {}
    data, _ = llm.chat_json(DOMAIN_SYSTEM, DOMAIN_PROMPT.format(profile=profile_summary_for_prompt(company_id, p, 900),
                                                                names="\n".join(f"- {n}" for n in names)),
                            purpose="competitor_domains", max_tokens=900)
    out = {}
    for d in (data or {}).get("names", []) if isinstance(data, dict) else []:
        if isinstance(d, dict) and d.get("name"):
            out[d["name"].strip().lower()] = d
    return out


def _llm_judge(company_id: int, p: dict, cands: list[dict]) -> dict[str, dict]:
    """One batched call; returns {domain: {verdict, why}}. Empty dict when no LLM is available."""
    if not cands:
        return {}
    listing = "\n\n".join(
        f"key={c['domain']} | name={c['name']}\n  title: {truncate((c.get('meta') or {}).get('title', ''), 120)}\n"
        f"  description: {truncate(c.get('description') or '', 220)}\n  homepage text: {truncate(c.get('homepage_text', ''), 480)}"
        for c in cands)
    verdicts: dict[str, dict] = {}
    for b in range(0, len(cands), 10):  # keep each request small for free-tier token limits
        chunk = listing.split("\n\n")[b:b + 10]
        data, used = llm.chat_json(JUDGE_SYSTEM, JUDGE_PROMPT.format(profile=profile_summary_for_prompt(company_id, p, 1400),
                                                                     cands="\n\n".join(chunk)),
                                   purpose="competitor_judge", max_tokens=1600)
        for v in (data or {}).get("verdicts", []) if isinstance(data, dict) else []:
            if isinstance(v, dict) and v.get("key") and str(v.get("verdict", "")).lower() in ("direct", "substitute", "adjacent", "unrelated"):
                verdicts[domain_of(v["key"]) or v["key"]] = {"verdict": v["verdict"].lower(), "why": clean_ws(v.get("why") or ""),
                                                            "by": used}
    return verdicts


def _homepage(url: str) -> tuple[str, dict, str]:
    r = fetcher().get(url)
    if not r.ok:
        return "", {}, r.error or f"HTTP {r.status}"
    return html_to_text(r.text, 20000), page_meta(r.text, r.final_url or url), ""


def _resolve_domain(name: str) -> str:
    """Confirm an official website for a competitor named without one (e.g. by the founder).
    Tries <name>.com/.in/.co/.io and accepts only if the page title or site name contains the name."""
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    if len(slug) < 3 or len(slug) > 30 or len(name.split()) > 4:  # sentences, not company names
        return ""
    name_toks = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 1]
    for tld in (".com", ".in", ".co", ".io", ".co.in"):
        dom = slug + tld
        r = fetcher().get(f"https://{dom}/")
        if not r.ok:
            continue
        m = page_meta(r.text, r.final_url or f"https://{dom}/")
        label = f"{m.get('title', '')} {m.get('og_site_name', '')}".lower()
        final = domain_of(r.final_url or dom)
        if not any(t in final.split(".")[0] for t in name_toks):  # redirected somewhere else (shortener, parked domain)
            continue
        if all(t in label.replace(" ", "") or t in label for t in name_toks):
            return final
    # Wikipedia: the article's external links usually include the official site.
    try:
        r = fetcher().get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3&srsearch="
                          + quote_plus(name), check_robots=False)
        hits = json.loads(r.text).get("query", {}).get("search", []) if r.ok else []
        for h in hits:
            if not all(t in h["title"].lower() for t in name_toks[:2]):
                continue
            er = fetcher().get("https://en.wikipedia.org/w/api.php?action=query&prop=extlinks&ellimit=60&format=json&titles="
                               + quote_plus(h["title"]), check_robots=False)
            pages = json.loads(er.text).get("query", {}).get("pages", {}) if er.ok else {}
            for pg in pages.values():
                for l in pg.get("extlinks", []):
                    d = domain_of(l.get("*", ""))
                    if d and any(t in d.split(".")[0] for t in name_toks) and not d.endswith(("wikipedia.org", "archive.org")):
                        return d
    except (ValueError, KeyError):
        pass
    return ""


def _shared_terms(a: str, b: str, n: int = 6) -> list[str]:
    ta, tb = tokens(a), set(tokens(b))
    seen, out = set(), []
    for t in ta:
        if t in tb and t not in seen and len(t) > 3:
            seen.add(t)
            out.append(t)
    from collections import Counter
    ca = Counter(t for t in ta if t in seen)
    return [t for t, _ in ca.most_common(n)]


def discover_competitors(company_id: int, step) -> list[dict]:
    p = get_profile(company_id)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    own_pages = db.q("SELECT text FROM research_pages WHERE company_id=? AND kind NOT IN ('news','wikipedia')", (company_id,))
    reference = "\n".join(r["text"] for r in own_pages)[:40000] + "\n" + " ".join(p.get("category_keywords", [])) * 3

    candidates: dict[str, dict] = {}
    by_name: dict[str, str] = {}

    def norm(n: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (n or "").lower())

    def add(name, domain, why, via, kind="direct", evidence_url=None):
        name = clean_ws(name)
        domain = domain_of(domain) if domain else ""
        if not name or domain == company["domain"] or norm(name) == norm(company["name"]):
            return
        key = domain or norm(name)
        existing = candidates.get(key) or candidates.get(by_name.get(norm(name), ""))
        if existing:
            existing["via"].add(via)
            if via == "founder":
                existing["why"] = (existing["why"] + " Named by the founder.").strip()
            if domain and not existing["domain"]:
                existing["domain"] = domain
            return
        candidates[key] = {"name": name, "domain": domain, "why": clean_ws(why or ""), "via": {via}, "kind": kind,
                           "evidence_url": evidence_url}
        by_name[norm(name)] = key

    for c in p.get("competitor_candidates", []):
        add(c.get("name"), c.get("domain"), c.get("why"), "research-llm", c.get("kind") or "direct")
    for name in (p.get("founder_named") or {}).get("competitors", []):
        add(name, "", "Named by the founder in the interview", "founder")

    # Competitors mentioned in category news (grounded in headlines).
    step("Mining category news for competitor mentions")
    headlines = []
    for kw in (p.get("category_keywords") or [])[:3]:
        res = adapters.bing_news({"query": kw, "type": "bing_news"})
        headlines += res.records[:12]
    if headlines:
        data, _ = llm.chat_json(COMP_NEWS_SYSTEM, COMP_NEWS_PROMPT.format(
            profile=profile_summary_for_prompt(company_id, p, 1500),
            headlines="\n".join(f"- {h['title']} — {truncate(h['snippet'], 160)} [{h['url']}]" for h in headlines[:36])),
            purpose="competitors_from_news")
        if isinstance(data, dict):
            blob = "\n".join(h["title"] + " " + h["snippet"] for h in headlines).lower()
            for c in data.get("competitors", [])[:10]:
                if isinstance(c, dict) and c.get("name") and c["name"].lower() in blob:  # must really appear
                    add(c["name"], c.get("domain"), c.get("why"), "category-news", "direct", c.get("evidence_url"))
        elif not p.get("competitor_candidates"):
            # No LLM: organisation-like names that recur across category headlines become candidates;
            # the website check + similarity verification below decides whether they are real competitors.
            from collections import Counter
            skip = {w.lower() for w in COUNTRIES} | {"the", "india", "indian", "new", "how", "why", "what", "top", "best"}
            cnt: Counter = Counter()
            first_url: dict[str, str] = {}
            for h in headlines:
                for m in re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+){0,2})\b", h["title"]):
                    if m.lower() in skip or norm(m) == norm(company["name"]) or len(m) < 4:
                        continue
                    cnt[m] += 1
                    first_url.setdefault(m, h["url"])
            for name, n in cnt.most_common(8):
                if n >= 2:
                    add(name, "", f"Recurs in {n} category headlines", "category-news", "direct", first_url[name])

    # Resolve + verify.
    step(f"Verifying {len(candidates)} competitor candidates against their live websites")
    verified, fetched = [], []
    ordered_cands = sorted(candidates.values(), key=lambda c: ("founder" not in c["via"], len(c["via"]) * -1))[:20]
    # Names without a domain (usually from the founder): ask the model for the official site, then verify it live.
    lookups = _llm_domains(company_id, p, [c["name"] for c in ordered_cands if not c["domain"]])
    substitutes = []
    for cand in ordered_cands:
        info = lookups.get(cand["name"].strip().lower()) or {}
        if not cand["domain"] and info.get("is_organisation") is False:
            cand.update(status="rejected", kind="substitute",
                        reason="Generic substitute named by the founder, not a single organisation — tracked through a "
                               "market news query instead of a competitor profile")
            substitutes.append({"name": cand["name"], "query": info.get("search_query") or cand["name"]})
            verified.append(cand)
            continue
        domain = cand["domain"] or domain_of(info.get("domain") or "") or _resolve_domain(cand["name"])
        if not domain:
            cand.update(status="rejected", reason="No official website could be confirmed - add it manually if relevant")
            verified.append(cand)
            continue
        cand["domain"] = domain
        url = f"https://{domain}/"
        text, meta, err = _homepage(url)
        if (err or len(text) < 200) and not domain.startswith("www."):
            url = f"https://www.{domain}/"
            text, meta, err = _homepage(url)
        if err or len(text) < 200:
            reachable = bool(err) and re.search(r"HTTP 40[13]|challenge|robots", err or "", re.I)
            if "founder" in cand["via"] and reachable:
                # The founder named it and the site exists but refuses automated reading: keep it, monitor via news.
                cand.update(status="approved", website=url, similarity=None, founder_unreadable=True,
                            reason=f"Named by the founder. Its homepage refuses automated reading ({err}); it exists, so it is "
                                   "monitored through news and app reviews instead of page-change tracking.")
            else:
                cand.update(status="rejected", website=url, reason=f"Homepage not readable ({err or 'too little text'})")
            verified.append(cand)
            continue
        cand.update(website=url, description=truncate(meta.get("description") or meta.get("title", ""), 300),
                    homepage_text=text[:8000], meta=meta)
        fetched.append(cand)
        verified.append(cand)
    if fetched:
        # One corpus (company + all candidates) so IDF down-weights generic web boilerplate.
        sims = sim_to_reference(reference, [c["homepage_text"] for c in fetched])
        for cand, sim in zip(fetched, sims):
            cand["similarity"] = round(sim, 3)
            cand["shared"] = _shared_terms(cand["homepage_text"], reference)

    ok = [c for c in verified if c.get("similarity") is not None]
    ok.sort(key=lambda c: (("founder" in c["via"]), c["similarity"]), reverse=True)
    max_sim = max((c["similarity"] for c in ok), default=0)
    step(f"Judging {len(ok)} fetched homepages against the company profile")
    verdicts = _llm_judge(company_id, p, ok)
    approved = sum(1 for c in verified if c.get("status") == "approved")
    for c in ok:
        v = verdicts.get(c["domain"]) or verdicts.get(domain_of(c.get("website") or ""))
        strong = c["similarity"] >= max(0.04, 0.3 * max_sim)
        evidence = (f"Live homepage similarity {c['similarity']:.2f} to {company['name']}'s site "
                    f"(shared terms: {', '.join(c['shared'][:5]) or 'few'}). Found via {', '.join(sorted(c['via']))}.")
        if v:
            c["kind"] = v["verdict"] if v["verdict"] != "unrelated" else c.get("kind")
            c["judge"] = v
            if v["verdict"] in ("direct", "substitute") and approved < 10:
                c["status"] = "approved"
                approved += 1
            elif v["verdict"] == "unrelated" and "founder" not in c["via"]:
                c["status"] = "rejected"
            else:
                c["status"] = "discovered"  # adjacent or over the cap: the founder decides in the Source Manager
            c["reason"] = f"{v['why'].rstrip('.')}. [homepage check: {v['verdict']}] {evidence}"
        else:  # no LLM available: lexical similarity alone decides
            if ("founder" in c["via"] or strong) and approved < 10:
                c["status"] = "approved"
                approved += 1
            else:
                c["status"] = "discovered"
            c["reason"] = f"{c['why'] or 'Offers a similar service'} Verified: {evidence}"
    if substitutes:
        prof = get_profile(company_id)
        prof["substitute_topics"] = substitutes
        from .profile import save_profile
        save_profile(company_id, prof, "Generic substitutes named by the founder")
    return verified


# --------------------------------------------------------------------------- probing
def _probe(company_id: int, cand: dict, ref_doc: str, keywords: set[str]) -> dict:
    src = {"id": -1, "type": cand["type"], "url": cand.get("url"), "query": cand.get("query"),
           "params": json.dumps(cand.get("params") or {}), "name": cand["name"]}
    if cand["type"] in ("webpage", "site_updates"):
        return {"records": 0, "newest_age_days": None, "status": "baseline", "sample": [], "keyword_share": None, "avg_sim": None}
    res = adapters.run_adapter(src)
    recs = res.records
    ages = [a for a in (age_days(r.get("published_at")) for r in recs) if a is not None]
    texts = [f"{r['title']} {r.get('snippet', '')}" for r in recs[:25]]
    sims = sim_to_reference(ref_doc, texts) if texts else []
    kw_share = (sum(1 for t in texts if keywords & set(tokens(t))) / len(texts)) if texts else 0.0
    return {
        "status": res.status, "error": res.error, "records": len(recs),
        "newest_age_days": round(min(ages), 1) if ages else None,
        "recent_30d": sum(1 for a in ages if a <= 30),
        "avg_sim": round(sum(sorted(sims, reverse=True)[:10]) / max(1, min(10, len(sims))), 3) if sims else 0.0,
        "keyword_share": round(kw_share, 2),
        "sample": [truncate(r["title"], 110) for r in recs[:3]],
    }


def _scores(cand: dict, probe: dict) -> tuple[float, float, float, float]:
    cred = cand.get("credibility") or credibility_for(cand["type"], cand.get("url", ""))
    if probe.get("status") == "baseline":
        spec, rec = cand.get("specific_hint", 0.8), 0.6
    else:
        if not probe.get("records"):
            return 0.0, 0.0, cred, round(0.25 * cred, 3)
        spec = min(1.0, 0.5 * probe.get("keyword_share", 0) + 3.0 * probe.get("avg_sim", 0))
        if cand.get("specific_hint"):
            spec = max(spec, cand["specific_hint"] * 0.85)
        a = probe.get("newest_age_days")
        rec = 1.0 if a is not None and a <= 7 else 0.8 if a is not None and a <= 30 else 0.5 if a is not None and a <= 90 else 0.2
    overall = 0.45 * spec + 0.30 * rec + 0.25 * cred
    return round(spec, 3), round(rec, 3), round(cred, 3), round(overall, 3)


def _reason(cand: dict, probe: dict) -> str:
    base = cand["why"].rstrip(".") + "."
    if probe.get("status") == "baseline":
        return base + " Baseline snapshot captured; future runs report what changed on the page."
    if probe.get("status") in ("needs_credentials",):
        return base + " " + (probe.get("error") or "")
    if not probe.get("records"):
        return base + f" Probe returned no records ({probe.get('error') or probe.get('status')}), so it is not approved."
    a = probe.get("newest_age_days")
    fresh = f"newest item {a:.0f} days old" if a is not None else "undated items"
    return (base + f" Probe: {probe['records']} records, {probe.get('recent_30d', 0)} in the last 30 days, {fresh}; "
            f"{int(100 * probe.get('keyword_share', 0))}% mention the company's problem-space terms. "
            f"e.g. “{probe['sample'][0] if probe.get('sample') else ''}”")


# --------------------------------------------------------------------------- candidate generation
def _candidates(company_id: int, comps: list[dict], step) -> list[dict]:
    p = get_profile(company_id)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    name = company["name"]
    cands: list[dict] = []

    def add(**kw):
        kw.setdefault("params", {})
        cands.append(kw)

    # ---------------- competitors
    for c in [c for c in comps if c["status"] == "approved"]:
        add(category="competitors", type="bing_news", name=f"{c['name']} — news", query=f'"{c["name"]}"',
            competitor=c, why=f"Tracks launches, pricing, partnerships, funding, hiring and expansion news for {c['name']} "
                              f"({c.get('kind', 'direct')} competitor: {truncate(c.get('why') or '', 140)})", specific_hint=0.75)
        if c.get("founder_unreadable") or not c.get("website"):
            continue  # site refuses automated reading: news (above) and app reviews only
        links = (c.get("meta") or {}).get("links", [])
        pricing = next((h for t, h in links if domain_of(h).endswith(c["domain"].replace("www.", "")) and
                        re.search(r"pricing|plans|fees|packages|premium", h + " " + t, re.I)), None)
        add(category="competitors", type="webpage", name=f"{c['name']} — {'pricing page' if pricing else 'homepage positioning'}",
            url=pricing or c["website"], competitor=c, specific_hint=0.85,
            why=f"Detects changes to {c['name']}'s {'pricing/plans' if pricing else 'positioning, offers and messaging'} on its own site (primary source)")
        blog = next((h for t, h in links if domain_of(h).endswith(c["domain"].replace("www.", "")) and
                     re.search(r"/(blog|news|press|insights|updates)/?$", urlparse(h).path, re.I)), None)
        if blog:
            add(category="competitors", type="site_updates", name=f"{c['name']} — blog/news", url=blog, competitor=c,
                specific_hint=0.8, why=f"New posts on {c['name']}'s own blog/newsroom reveal product launches and positioning shifts")

    # competitor & own apps -> customers
    own_apps = p.get("apps") or []
    for a in own_apps[:1]:
        add(category="customers", type="appstore_reviews", name=f"{name} iOS app reviews", url=a["url"],
            params={"app_id": a["app_id"], "country": a["country"], "app_name": a["name"], "app_url": a["url"]},
            why=f"Actual customer language about {name}'s own app (ratings, complaints, requests)", specific_hint=0.9)
    step("Looking for competitor apps (customer reviews of alternatives)")
    napps = 0
    for c in [c for c in comps if c["status"] == "approved"][:6]:
        if napps >= 3:
            break
        for cc in ("in", "us"):
            hits = adapters.itunes_search(c["name"], cc, 4)
            match = next((h for h in hits if c["domain"].split(".")[0] in (h.get("sellerUrl") or "").lower()
                          or c["name"].lower().replace(" ", "") in (h.get("sellerName") or "").lower().replace(" ", "")
                          or c["name"].lower().replace(" ", "") in (h.get("trackName") or "").lower().replace(" ", "")), None)
            if match:
                add(category="customers", type="appstore_reviews", name=f"{c['name']} app reviews (alternative)",
                    url=match.get("trackViewUrl"), competitor=c,
                    params={"app_id": match["trackId"], "country": cc, "app_name": match.get("trackName"),
                            "app_url": match.get("trackViewUrl")},
                    why=f"Complaints and praise for {c['name']}, an alternative customers consider — reveals unmet needs {name} can win on",
                    specific_hint=0.7)
                napps += 1
                break

    # ---------------- customers: where problems are discussed
    for qtext in (p.get("customer_problem_queries") or [])[:4]:
        add(category="customers", type="bing_news", name=f"Customer problem: {qtext}", query=qtext,
            why=f"Public reporting/discussion of the problem customers have before choosing {name} ('{qtext}')")
    # Stack Exchange communities with real hits for this company's problem queries.
    step("Testing Q&A communities for real customer questions")
    se_queries = [q for q in (p.get("category_keywords") or [])[:3]]
    se_hits = []
    for site in SE_SITES:
        q0 = se_queries[0] if se_queries else name
        r = fetcher().get(f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=creation&pagesize=10&q={quote_plus(q0)}&site={site}",
                          check_robots=False)
        if not r.ok:
            continue
        try:
            items = json.loads(r.text).get("items", [])
        except ValueError:
            continue
        recent = [i for i in items if age_days(db_iso(i.get("creation_date"))) is not None and age_days(db_iso(i.get("creation_date"))) < 365]
        if len(recent) >= 2:
            se_hits.append((len(recent), site, q0))
    for n, site, q0 in sorted(se_hits, reverse=True)[:2]:
        add(category="customers", type="stackexchange", name=f"{site}.stackexchange — '{q0}'", query=q0,
            url=f"https://{site}.stackexchange.com/search?q={quote_plus(q0)}", params={"site": site},
            why=f"Real questions people ask in the {site} community about '{q0}' ({n} in the last year) — unfiltered customer language")
    # Hacker News only if the problem space is actually discussed there.
    kw0 = (p.get("category_keywords") or [name])[0]
    add(category="customers", type="hn", name=f"Hacker News — '{kw0}'", query=kw0,
        url=f"https://hn.algolia.com/?q={quote_plus(kw0)}",
        why=f"Practitioner/early-adopter discussion of '{kw0}' (kept only if the probe shows recent, relevant threads)")
    # Reddit: identified, but only usable via the official API.
    for sub in _suggest_subreddits(company_id, p)[:3]:
        add(category="customers", type="reddit", name=f"r/{sub}", url=f"https://www.reddit.com/r/{sub}/", params={"subreddit": sub},
            why=f"Community where {name}'s customers discuss their problems", blocked=True)

    # ---------------- market
    for t in (p.get("market_topics") or [])[:5]:
        add(category="market", type="bing_news", name=f"Market: {t}", query=t,
            why=f"Industry/market development that can change demand or economics for {name}: '{t}'")
    for s in (p.get("substitute_topics") or [])[:3]:
        add(category="competitors", type="bing_news", name=f"Substitute: {s['name']}", query=s["query"],
            why=f"The founder named '{s['name']}' as a substitute customers choose instead of {name}; it is a category, "
                f"not one company, so it is tracked as a news topic", specific_hint=0.6)
    for t in (p.get("regulatory_topics") or [])[:4]:
        add(category="market", type="bing_news", name=f"Regulation: {t}", query=t, credibility=0.75,
            why=f"Regulatory/policy development directly affecting {name}'s customers or operations: '{t}'")
    geos = set(p.get("geographies") or [])
    reg_q = " OR ".join(f'"{t}"' for t in (p.get("regulatory_topics") or [])[:1]) or (p.get("category_keywords") or [name])[0]
    if geos & {"UK", "United Kingdom"}:
        kw = (p.get("regulatory_topics") or p.get("category_keywords") or [name])[0]
        short = " ".join(kw.split()[:3])
        add(category="market", type="rss", name=f"GOV.UK announcements — '{short}'",
            url=f"https://www.gov.uk/search/news-and-communications.atom?keywords={quote_plus(short)}",
            why="Official UK government announcements/policy papers (primary source) matching the company's regulatory topics",
            credibility=0.95)
    if geos & {"USA", "US", "United States"}:
        kw = (p.get("regulatory_topics") or p.get("category_keywords") or [name])[0]
        short = " ".join(kw.split()[:3])
        add(category="market", type="rss", name=f"US Federal Register — '{short}'",
            url=f"https://www.federalregister.gov/api/v1/documents.rss?conditions%5Bterm%5D={quote_plus(short)}&order=newest",
            why="Official US federal rules and notices (primary source) matching the company's regulatory topics", credibility=0.95)
    add(category="market", type="gdelt", name=f"GDELT global news — {truncate(reg_q, 60)}", query=reg_q,
        params={"timespan": "30d"}, why="Global news index (100+ countries) to catch regulatory/market developments mainstream feeds miss")
    step("Discovering feeds of trade publications and regulators")
    for pub in (p.get("publications") or [])[:8]:
        dom = domain_of(pub.get("domain", ""))
        if not dom:
            continue
        feed = adapters.discover_feed(f"https://{dom}/")
        if feed:
            add(category="market", type="rss", name=f"{pub.get('name') or dom} (feed)", url=feed,
                why=f"{pub.get('why') or 'Specialist publication for this industry'}", credibility=0.95 if ("." + dom).endswith(GOV_SUFFIXES) else 0.8)

    # ---------------- internal
    add(category="internal", type="upload", name="Internal uploads inbox (CRM, calls, support, feedback)",
        url=None, why="CSV / Excel / PDF / text uploads (CRM exports, sales & meeting transcripts, support conversations, "
                      "product feedback) are parsed into records to find objections, lost-deal reasons and product requests",
        internal=True)
    return cands


def db_iso(ts):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


SUBREDDIT_SYSTEM = "You know online communities well. Suggest real, active subreddits only."
SUBREDDIT_PROMPT = """{profile}

Which 3 existing subreddits do this company's customers use to discuss the problems it solves?
Return JSON: {{"subreddits": ["name without r/"]}}"""


def _suggest_subreddits(company_id: int, p: dict) -> list[str]:
    data, _ = llm.chat_json(SUBREDDIT_SYSTEM, SUBREDDIT_PROMPT.format(profile=profile_summary_for_prompt(company_id, p, 1200)),
                            purpose="subreddits", max_tokens=300)
    subs = []
    if isinstance(data, dict):
        for s in data.get("subreddits", []):
            s = re.sub(r"^/?r/", "", str(s)).strip()
            if re.fullmatch(r"[A-Za-z0-9_]{3,21}", s):
                subs.append(s)
    return subs


# --------------------------------------------------------------------------- build
def _source_key(c: dict) -> str:
    return f"{c['type']}|{(c.get('url') or '').lower()}|{(c.get('query') or '').lower()}|{json.dumps(c.get('params') or {}, sort_keys=True)}"


def build_signal_stack(company_id: int, ctx: JobContext | None = None, origin: str = "discovery") -> dict:
    step = ctx.step if ctx else (lambda m: None)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    p = get_profile(company_id)

    step("Discovering and verifying competitors")
    comps = discover_competitors(company_id, step)
    existing_comp = {(r["domain"] or r["name"].lower()): r for r in db.q("SELECT * FROM competitors WHERE company_id=?", (company_id,))}
    for c in comps:
        key = c.get("domain") or c["name"].lower()
        row = {"company_id": company_id, "name": c["name"], "website": c.get("website"), "domain": c.get("domain"),
               "description": c.get("description"), "why": c.get("reason"), "kind": c.get("kind"),
               "similarity": c.get("similarity"), "evidence_url": c.get("evidence_url") or c.get("website"),
               "evidence": ", ".join(c.get("shared", [])), "discovered_via": ", ".join(sorted(c["via"])),
               "status": c["status"], "created_at": db.now()}
        if key in existing_comp:
            keep_status = existing_comp[key]["status"] if existing_comp[key]["status"] in ("approved", "rejected") else c["status"]
            row["status"] = keep_status
            row.pop("created_at")
            db.update("competitors", existing_comp[key]["id"], row)
            c["id"] = existing_comp[key]["id"]
        else:
            c["id"] = db.insert("competitors", row)
        c["status"] = row["status"]

    # Keep the stack focused: at most 10 approved competitors across discovery runs. Founder-named ones stay; the
    # weakest (lowest homepage similarity) of the rest go back to 'discovered' for the founder to review.
    appr = db.q("SELECT id, name, similarity, discovered_via FROM competitors WHERE company_id=? AND status='approved'",
                (company_id,))
    if len(appr) > 10:
        appr.sort(key=lambda r: ("founder" in (r["discovered_via"] or ""), r["similarity"] or 0), reverse=True)
        for r in appr[10:]:
            db.update("competitors", r["id"], {"status": "discovered"})
            for c in comps:
                if c.get("id") == r["id"]:
                    c["status"] = "discovered"
        step(f"Competitor cap: {len(appr) - 10} lower-similarity competitor(s) moved to 'discovered' for founder review")

    ref_doc = interest_document(company_id, p)
    keywords = set(tokens(" ".join(p.get("category_keywords", []) + p.get("offerings", [])[:5] + p.get("geographies", []))))
    cands = _candidates(company_id, comps, step)
    existing = {(_source_key({"type": s["type"], "url": s["url"], "query": s["query"], "params": db.loads(s["params"], {})})): s
                for s in db.q("SELECT * FROM sources WHERE company_id=?", (company_id,))}
    added, approved = 0, 0
    step(f"Probing {len(cands)} candidate sources (fetching each once to score specificity, recency, credibility)")
    for i, c in enumerate(cands, 1):
        key = _source_key(c)
        if key in existing:
            continue
        if c.get("internal"):
            probe, spec, rec, cred, overall, status = {"status": "internal"}, 1.0, 1.0, 0.8, 0.9, "approved"
        elif c.get("blocked"):
            probe = {"status": "needs_credentials", "error": "Reddit forbids anonymous crawling (robots.txt + Public Content "
                                                             "Policy). Enable with free official API credentials."}
            spec, rec, cred, overall, status = 0.8, 0.0, credibility_for("reddit"), 0.0, "blocked"
        else:
            step(f"[{i}/{len(cands)}] probing {c['name']}")
            probe = _probe(company_id, c, ref_doc, keywords)
            spec, rec, cred, overall = _scores(c, probe)
            if probe.get("status") in ("robots_blocked", "challenge"):
                status = "blocked"
            elif c["type"] == "hn" and (probe.get("recent_30d", 0) < 3 or probe.get("keyword_share", 0) < 0.3):
                status = "rejected"
            elif overall >= config.AUTO_APPROVE_THRESHOLD and c.get("competitor", {}).get("status", "approved") == "approved":
                status = "approved"
            elif probe.get("records", 0) == 0 and probe.get("status") != "baseline":
                status = "rejected"
            else:
                status = "discovered"
        reason = c["why"] if c.get("internal") else _reason(c, probe)
        sid = db.insert("sources", {
            "company_id": company_id, "category": c["category"], "type": c["type"], "name": c["name"],
            "url": c.get("url") or (adapters.bing_news_url(c["query"]) if c["type"] == "bing_news" else None),
            "query": c.get("query"), "params": c.get("params") or {}, "competitor_id": (c.get("competitor") or {}).get("id"),
            "reason": reason, "specific_score": spec, "recency_score": rec, "credibility_score": cred, "overall_score": overall,
            "probe": probe, "status": status, "origin": origin, "created_at": db.now(),
            "approved_at": db.now() if status == "approved" else None})
        added += 1
        approved += status == "approved"
        if c["type"] in ("webpage", "site_updates"):
            # baseline snapshot so the first monitoring run can already detect change
            src = db.one("SELECT * FROM sources WHERE id=?", (sid,))
            try:
                adapters.run_adapter(src)
            except Exception:  # noqa: BLE001
                pass
    db.update("companies", company_id, {"status": "stack_ready", "updated_at": db.now()})
    summary = {"competitors_found": len(comps), "competitors_approved": sum(c["status"] == "approved" for c in comps),
               "sources_added": added, "sources_auto_approved": approved}
    step(f"Signal Stack built: {summary}")
    return summary
