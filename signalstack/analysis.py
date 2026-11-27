"""Intelligence analysis and ranking.

Three measures are kept deliberately separate (and never merged with the AI's confidence):
    credibility        - how reliable the *source* is (publisher / source-type prior)
    evidence_strength  - how well the *claim* is evidenced (primary source, corroboration)
    relevance          - how strategically relevant it is to *this* company (TF-IDF vs profile + entities)
The rank score combines relevance, impact, urgency, credibility, recency and novelty, and every
item stores a plain-English explanation of why it ranked where it did.

The LLM (if available) adds interpretation for the most important items: what happened (restricted
to the evidence), why it matters for this company, business area, opportunity/threat, urgency and what
to consider doing. Interpretation is stored separately from evidence and labelled as such.
"""
from __future__ import annotations

import json
import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config, db, llm
from .jobs import JobContext
from .profile import FIELD_LABEL, get_profile, interest_document, profile_summary_for_prompt
from .textutil import STOP, age_days, domain_of, tokens, truncate

_vader = SentimentIntensityAnalyzer()

TOPICS: list[tuple[str, float, str]] = [
    # (topic, impact weight, regex)
    ("regulation", 0.9, r"\b(regulat\w*|policy|polic(y|ies)|law|bill|act|rule[sd]?|guideline|compliance|ban(ned|s)?|cap(ped|s)?|"
                        r"visa|permit|immigration|government|ministry|minister|court|ruling|sebi|rbi|ugc|fda|gdpr|tax|tariff|"
                        r"license|licence|mandate|parliament|senate|congress)\b"),
    ("pricing", 0.85, r"\b(pric(e|es|ing)|fee[s]?|discount|cheaper|subscription|plan[s]?|cost[s]?|tariff|free tier|"
                      r"commission|refund|offer[s]?)\b"),
    ("acquisition", 0.85, r"\b(acquir\w*|acquisition|merger|merge[sd]?|buyout|takeover|consolidat\w*)\b"),
    ("product_launch", 0.8, r"\b(launch\w*|unveil\w*|introduc\w*|rolls? out|new (feature|product|service|app|platform|programme|program)|"
                            r"releas\w*|debut\w*)\b"),
    ("funding", 0.75, r"\b(rais(es|ed|ing)|funding|series [a-f]|seed round|invest\w*|valuation|ipo|unicorn|backed)\b"),
    ("partnership", 0.7, r"\b(partner\w*|tie[- ]?up|collaborat\w*|alliance|mou|joint venture|teams up)\b"),
    ("expansion", 0.7, r"\b(expan\w*|enters?|new (office|market|campus|city|country)|opens?|international|global)\b"),
    ("people", 0.55, r"\b(hir(es|ed|ing)|layoff\w*|appoint\w*|ceo|cto|cfo|founder|resign\w*|job cuts|headcount)\b"),
    ("complaint", 0.7, r"\b(complain\w*|scam|fraud|worst|terrible|refund|cheat\w*|poor|bad service|disappoint\w*|issue[s]?|"
                       r"problem[s]?|not working|crash\w*|rejected|delay\w*)\b"),
    ("demand_data", 0.65, r"\b(enrol\w*|admission\w*|applications?|demand|surge|decline|drop|fall|rise|record|survey|report|"
                          r"data|statistics|percent|%|growth|market size)\b"),
    ("technology", 0.6, r"\b(ai|artificial intelligence|genai|chatgpt|llm|automation|platform|digital|app)\b"),
]
TOPIC_IMPACT = {t: w for t, w, _ in TOPICS}
TOPIC_RX = [(t, re.compile(rx, re.I)) for t, _, rx in TOPICS]

THREAT_RX = re.compile(r"\b(ban|cap|restrict\w*|tighten\w*|crackdown|decline\w*|drop\w*|fall\w*|falls|cut\w*|halt\w*|"
                       r"suspend\w*|reject\w*|scam|fraud|lawsuit|fine[ds]?|penalt\w*|layoff\w*|loss\w*|delay\w*|"
                       r"shortage|risk\w*|warn\w*|slump|stricter|hike[sd]?|deport\w*|revok\w*|complain\w*|worst|poor)\b", re.I)
OPP_RX = re.compile(r"\b(growth|grow\w*|surge|rise|rising|record|boost\w*|new route|eas(e|es|ed|ing)|relax\w*|scholarship\w*|"
                    r"opportunit\w*|demand|expand\w*|increase\w*|launch\w*|partner\w*|funding|subsid\w*|incentive\w*|"
                    r"open(s|ed|ing)?|welcome\w*|wish|would love|please add|need)\b", re.I)
URGENT_RX = re.compile(r"\b(effective (immediately|from)|from (january|february|march|april|may|june|july|august|september|"
                       r"october|november|december)|deadline|immediately|with immediate effect|starting (today|next)|"
                       r"new rules?|announced today|breaking|urgent|this week|ban|suspend\w*)\b", re.I)

# Widely-recognised news publishers (credibility prior by publisher; generic, not per company).
TIER1 = ("reuters", "bloomberg", "ft.com", "financial times", "wsj", "wall street journal", "nytimes", "new york times",
         "bbc", "theguardian", "guardian", "economist", "apnews", "associated press", "cnbc", "forbes", "fortune",
         "economictimes", "economic times", "livemint", "mint", "business-standard", "business standard", "thehindu",
         "the hindu", "hindustantimes", "hindustan times", "indianexpress", "indian express", "timesofindia",
         "times of india", "moneycontrol", "ndtv", "indiatoday", "india today", "techcrunch", "theverge", "wired",
         "inc42", "yourstory", "entrackr", "the ken", "thepienews", "pie news", "icef", "university world news",
         "insidehighered", "times higher education", "timeshighereducation", "cnn", "washington post", "axios",
         "the print", "theprint", "scroll", "deccan herald", "financial express", "businessline", "the telegraph",
         "abc news", "cbc", "globe and mail", "sydney morning herald", "straits times", "south china morning post")

OWNERS = {"regulation": "Founder + Compliance", "pricing": "Founder + Growth", "acquisition": "Founder/CEO",
          "product_launch": "Product lead", "funding": "Founder/CEO", "partnership": "Partnerships/BD",
          "expansion": "Growth lead", "people": "Founder + HR", "complaint": "Customer success / Ops",
          "demand_data": "Marketing lead", "technology": "Product lead", "other": "Founder"}


def classify_topic(text: str) -> str:
    best, best_n = "other", 0
    for t, rx in TOPIC_RX:
        n = len(rx.findall(text))
        if n > best_n:
            best, best_n = t, n
    return best


def publisher_credibility(source_cred: float, publisher: str, url: str, stype: str) -> float:
    d = "." + domain_of(url)
    if d.endswith((".gov", ".gov.uk", ".gc.ca", ".gov.au", ".gov.in", ".nic.in", ".europa.eu", ".gov.sg")):
        return 0.95
    if stype in ("bing_news", "gdelt", "rss"):
        pub = f"{publisher} {d}".lower()
        if any(t in pub for t in TIER1):
            return max(source_cred, 0.85)
        return min(source_cred, 0.6)
    return source_cred


def verification_for(stype: str, url: str, item: dict, credibility: float) -> str:
    d = "." + domain_of(url)
    if stype in ("webpage", "site_updates"):
        return "primary_source"
    if d.endswith((".gov", ".gov.uk", ".gc.ca", ".gov.au", ".gov.in", ".nic.in", ".europa.eu")):
        return "primary_source"
    if stype == "upload":
        return "internal"
    if stype in ("appstore_reviews", "stackexchange", "hn", "reddit"):
        return "user_generated"
    if item.get("corroborations", 0) >= 1:
        return "corroborated"
    return "reported" if credibility >= 0.7 else "unverified"


EVIDENCE = {"primary_source": 0.9, "corroborated": 0.8, "internal": 0.7, "reported": 0.6, "user_generated": 0.45,
            "unverified": 0.3}


def _entity_hits(text: str, names: list[str]) -> list[str]:
    low = text.lower()
    return [n for n in names if n and len(n) > 2 and re.search(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])", low)]


def score_items(company_id: int, item_ids: list[int] | None = None, ctx: JobContext | None = None) -> int:
    """Deterministic scoring for all (or given) items. Cheap; re-run after profile changes."""
    step = ctx.step if ctx else (lambda m: None)
    p = get_profile(company_id)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    ref = interest_document(company_id, p)
    comps = db.q("SELECT id, name FROM competitors WHERE company_id=? AND status='approved'", (company_id,))
    comp_names = [c["name"] for c in comps]
    geos = [g for g in p.get("geographies", []) if len(g) > 1]
    kw_terms = set(tokens(" ".join(p.get("category_keywords", []))))
    sources = {s["id"]: s for s in db.q("SELECT * FROM sources WHERE company_id=?", (company_id,))}
    items = db.q("SELECT * FROM items WHERE company_id=? ORDER BY id", (company_id,))
    if not items:
        return 0
    texts = [f"{i['title']}. {i['snippet'] or ''}" for i in items]
    # Facets: short, specific statements of what this company cares about. Matching a headline against each facet is
    # far more discriminative than matching it against one long profile document.
    facets: list[str] = []
    for k in ("category_keywords", "offerings", "customer_segments", "market_topics", "regulatory_topics",
              "customer_problem_queries", "value_props"):
        facets += [str(x) for x in p.get(k, []) or [] if str(x).strip()]
    for vals in (p.get("intelligence") or {}).values():
        facets += [v.get("text", "") if isinstance(v, dict) else str(v) for v in vals or []]
    facets = [f for f in dict.fromkeys(facets) if len(tokens(f)) >= 1] or [ref[:500]]
    vec = TfidfVectorizer(stop_words=list(STOP), ngram_range=(1, 2), sublinear_tf=True, max_features=60000, min_df=1)
    X = vec.fit_transform([ref, *texts, *facets])
    n_items = len(texts)
    Xi, Xf = X[1:1 + n_items], X[1 + n_items:]
    rel_raw = cosine_similarity(X[0], Xi).ravel()
    facet_sims = cosine_similarity(Xi, Xf)
    item_sims = cosine_similarity(Xi)
    targets = set(item_ids) if item_ids else None
    n = 0
    for idx, it in enumerate(items):
        if targets is not None and it["id"] not in targets:
            continue
        src = sources.get(it["source_id"]) or {"type": "rss", "credibility_score": 0.5, "category": it["category"]}
        text = texts[idx]
        # --- relevance (strategic, company-specific)
        hits_company = _entity_hits(text, [company["name"], company["domain"].split(".")[0]])
        hits_comp = _entity_hits(text, comp_names)
        hits_geo = _entity_hits(text, geos)
        kw_overlap = len(kw_terms & set(tokens(text))) / max(1, min(6, len(kw_terms)))
        fs = sorted(facet_sims[idx].tolist(), reverse=True)
        top1, top3 = fs[0], sum(fs[:3]) / max(1, len(fs[:3]))
        best_facet = facets[int(facet_sims[idx].argmax())]
        rel = min(1.0, 0.9 * top1 + 0.6 * top3 + 1.5 * float(rel_raw[idx]) + 0.15 * min(1.0, kw_overlap)
                  + (0.3 if hits_company else 0) + (0.25 if hits_comp or it["competitor_id"] else 0)
                  + (0.08 if hits_geo else 0) + (0.25 if it["category"] == "internal" else 0))
        # --- topic, sentiment
        topic = classify_topic(text)
        sent = _vader.polarity_scores(text)["compound"]
        # --- recency / novelty
        a = age_days(it["published_at"])
        recency = math.exp(-a / 10) if a is not None else 0.45
        earlier = [j for j in range(idx) if items[j]["id"] != it["dup_of"] and items[j]["dup_of"] != it["id"]]
        novelty = 1.0 - (max(float(item_sims[idx, j]) for j in earlier) if earlier else 0.0)
        novelty = max(0.0, min(1.0, novelty))
        if it["dup_of"]:
            novelty = min(novelty, 0.2)
        # --- credibility / evidence (independent of AI)
        cred = publisher_credibility(src.get("credibility_score") or 0.5, it["publisher"] or "", it["url"] or "", src["type"])
        verification = verification_for(src["type"], it["url"] or "", it, cred)
        evidence = min(1.0, EVIDENCE[verification] + 0.1 * min(2, it["corroborations"] or 0))
        # --- impact / urgency / polarity
        impact = TOPIC_IMPACT.get(topic, 0.4) * (0.45 + 0.55 * rel)
        threat_n, opp_n = len(THREAT_RX.findall(text)), len(OPP_RX.findall(text))
        urgency = 0.25 + 0.3 * recency + (0.25 if URGENT_RX.search(text) else 0) + (0.1 if topic == "regulation" else 0) \
            + (0.1 if sent < -0.4 else 0)
        urgency = min(1.0, urgency)
        if src["category"] == "competitors" and topic in ("product_launch", "funding", "pricing", "expansion", "partnership", "acquisition"):
            polarity = "threat"
        elif src["category"] == "customers" and it["competitor_id"] and (sent < -0.2 or topic == "complaint"):
            polarity = "opportunity"  # customers unhappy with an alternative
        elif threat_n > opp_n or (sent < -0.35 and threat_n):
            polarity = "threat"
        elif opp_n > threat_n:
            polarity = "opportunity"
        else:
            polarity = "neutral"
        base = 0.30 * rel + 0.22 * impact + 0.15 * urgency + 0.11 * cred + 0.11 * recency + 0.11 * novelty
        score = base * (0.35 + 0.65 * min(1.0, rel / 0.5))  # relevance gate: off-topic items cannot rank high
        significant = int(rel >= 0.4 and score >= 0.38 and not it["dup_of"])
        why = []
        if top1 >= 0.15:
            why.append(f"matches profile facet “{truncate(best_facet, 60)}”")
        if hits_company:
            why.append(f"mentions {company['name']}")
        if hits_comp:
            why.append(f"mentions competitor {', '.join(hits_comp[:2])}")
        if hits_geo:
            why.append(f"concerns {', '.join(hits_geo[:3])}")
        why.append(f"relevance {rel:.2f}")
        why.append(f"{topic.replace('_', ' ')} topic (impact {impact:.2f})")
        why.append(f"{'published ' + str(int(a)) + 'd ago' if a is not None else 'undated'}")
        why.append(f"source reliability {cred:.2f} ({verification.replace('_', ' ')})")
        why.append(f"novelty {novelty:.2f}")
        if it["corroborations"]:
            why.append(f"corroborated by {it['corroborations']} other publisher(s)")
        db.update("items", it["id"], {
            "topic": topic, "sentiment": round(sent, 3), "relevance": round(rel, 3), "impact": round(impact, 3),
            "urgency": round(urgency, 3), "credibility": round(cred, 3), "recency": round(recency, 3),
            "novelty": round(novelty, 3), "evidence_strength": round(evidence, 3), "score": round(score, 3),
            "polarity": polarity if not (db.loads(it["analysis"], {}) or {}).get("interpretation") else it["polarity"] or polarity,
            "verification": verification, "significant": significant if not it["analysed_by"] else it["significant"],
            "explanation": "; ".join(why)})
        n += 1
    step(f"Scored {n} records (relevance, impact, urgency, credibility, recency, novelty)")
    return n


INTERPRET_SYSTEM = """You are the head of strategy at the company described below, triaging this week's intelligence.
For each item:
- "what_happened": ONLY what the evidence text says (no outside facts). If the snippet is thin, say so.
- "material": true only if it could plausibly change this company's revenue, customers, costs, positioning or compliance;
  false for general news that merely shares keywords.
- "why_it_matters": specific to THIS company - name the affected offering, segment, geography or founder priority. No generic advice.
- "business_area": one of: product, pricing, sales, marketing, partnerships, operations, compliance, customer_success, strategy.
- "polarity": opportunity | threat | neutral.  "urgency": high | medium | low.
- "consider": one concrete thing the company should consider doing (verb first, specific, small enough for this week or month).
- "confidence": 0-1, your confidence in the interpretation (this is NOT evidence that the event is true)."""

INTERPRET_PROMPT = """Company profile (from research + founder interview):
{profile}

Items:
{items}

Return JSON: {{"items": [{{"id": <id>, "what_happened": "...", "material": true, "why_it_matters": "...", "business_area": "...",
"polarity": "...", "urgency": "...", "consider": "...", "confidence": 0.0}}]}}"""

URG = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _float_or_none(v):
    try:
        f = float(v)
        return f if 0 <= f <= 1 else None
    except (TypeError, ValueError):
        return None


def interpret_items(company_id: int, limit: int | None = None, ctx: JobContext | None = None, since: str | None = None) -> int:
    """LLM interpretation for the highest-ranked, not-yet-interpreted items (rule-based fallback)."""
    step = ctx.step if ctx else (lambda m: None)
    p = get_profile(company_id)
    prof = profile_summary_for_prompt(company_id, p, 2600)
    limit = limit or config.INTERPRET_LIMIT
    # Lexical relevance is only a coarse pre-filter (headlines are short); the model judges materiality.
    where = "company_id=? AND dup_of IS NULL AND analysed_by IS NULL AND relevance >= 0.15"
    params: list = [company_id]
    if since:
        where += " AND discovered_at >= ?"
        params.append(since)
    cands = db.q(f"SELECT * FROM items WHERE {where} ORDER BY score DESC LIMIT ?", (*params, limit))
    if not cands:
        return 0
    step(f"Interpreting top {len(cands)} signals with {llm.active_provider_label()}")
    done = 0
    for b in range(0, len(cands), 8):
        batch = cands[b:b + 8]
        listing = "\n\n".join(
            f"id={i['id']} | category={i['category']} | topic={i['topic']} | publisher={i['publisher']} | date={i['published_at'] or 'unknown'}\n"
            f"title: {i['title']}\nevidence: {truncate(i['snippet'] or '(headline only)', 420)}" for i in batch)
        data, used = llm.chat_json(INTERPRET_SYSTEM, INTERPRET_PROMPT.format(profile=prof, items=listing),
                                   purpose="interpret", temperature=0.2, max_tokens=3500)
        by_id = {}
        if isinstance(data, dict):
            for d in data.get("items", []):
                try:
                    by_id[int(d.get("id"))] = d
                except (TypeError, ValueError):
                    continue
        for it in batch:
            d = by_id.get(it["id"])
            if d:
                urg = URG.get(str(d.get("urgency", "")).lower(), it["urgency"] or 0.5)
                material = bool(d.get("material", True))
                analysis = {"evidence": {"title": it["title"], "snippet": it["snippet"], "url": it["url"],
                                         "publisher": it["publisher"], "published_at": it["published_at"]},
                            "interpretation": {k: d.get(k) for k in ("what_happened", "why_it_matters", "business_area",
                                                                    "polarity", "urgency", "consider", "material")},
                            "interpreted_by": used}
                pol = str(d.get("polarity", it["polarity"] or "neutral")).lower()
                if pol not in ("opportunity", "threat", "neutral"):
                    pol = it["polarity"] or "neutral"
                db.update("items", it["id"], {
                    "analysis": analysis, "polarity": pol, "urgency": round(0.5 * (it["urgency"] or 0.5) + 0.5 * urg, 3),
                    "ai_confidence": _float_or_none(d.get("confidence")),
                    # material per the analyst AND not off-topic per the independent lexical measure
                    "significant": int(material and (it["relevance"] or 0) >= 0.2 and (it["score"] or 0) >= 0.2),
                    "analysed_by": used, "analysed_at": db.now()})
            else:
                db.update("items", it["id"], {"analysis": _rule_interpretation(company_id, it, p), "analysed_by": "rule-based",
                                              "analysed_at": db.now()})
            done += 1
    step(f"Interpreted {done} signals")
    return done


def _rule_interpretation(company_id: int, it: dict, p: dict) -> dict:
    text = f"{it['title']} {it['snippet'] or ''}"
    matched = [o for o in (p.get("offerings", []) + p.get("category_keywords", []) + p.get("geographies", []))
               if o and set(tokens(o)) & set(tokens(text))][:3]
    prio = [v.get("text") for v in (p.get("intelligence", {}).get("growth_priorities") or []) if isinstance(v, dict)][:1]
    topic = it["topic"] or "other"
    area = {"regulation": "compliance", "pricing": "pricing", "product_launch": "product", "funding": "strategy",
            "partnership": "partnerships", "expansion": "strategy", "complaint": "customer_success",
            "demand_data": "marketing", "people": "operations", "technology": "product"}.get(topic, "strategy")
    why = (f"Touches {', '.join(matched)}" if matched else "Shares vocabulary with the company's problem space") + \
          (f" and the founder's priority '{truncate(prio[0], 80)}'." if prio else ".")
    consider = {
        "regulation": "Brief the team on the change and update customer-facing guidance where it applies.",
        "pricing": "Compare this pricing move with your current packages before your next pricing review.",
        "product_launch": "Assess whether the launch overlaps your offering and prepare a positioning response.",
        "funding": "Expect more aggressive marketing from this player; monitor their acquisition channels.",
        "partnership": "Check whether a similar partnership is open to you.",
        "complaint": "Check whether your customers report the same issue and address it in your messaging.",
        "demand_data": "Use the data point in sales/marketing content and revisit demand forecasts.",
    }.get(topic, "Keep monitoring; no action needed yet.")
    return {"evidence": {"title": it["title"], "snippet": it["snippet"], "url": it["url"], "publisher": it["publisher"],
                         "published_at": it["published_at"]},
            "interpretation": {"what_happened": it["title"], "why_it_matters": why, "business_area": area,
                               "polarity": it["polarity"], "urgency": "high" if (it["urgency"] or 0) > 0.7 else "medium" if (it["urgency"] or 0) > 0.45 else "low",
                               "consider": consider, "material": bool(it["significant"])},
            "interpreted_by": "rule-based"}


def analyse(company_id: int, ctx: JobContext | None = None, since: str | None = None) -> None:
    score_items(company_id, ctx=ctx)
    interpret_items(company_id, ctx=ctx, since=since)
