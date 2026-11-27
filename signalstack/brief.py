"""Monday Morning Intelligence Brief.

Built in two layers:
1. Evidence layer (deterministic): selects ranked, significant, de-duplicated signals from the
   period, per category, with links, dates, snippets and the separate reliability / evidence /
   relevance measures. Empty categories are stated explicitly - nothing is manufactured.
2. Interpretation layer (LLM if available, rule-based otherwise): executive summary, opportunities,
   risks, actions and watch-list. Every interpretive statement must cite signal IDs that exist in the
   evidence layer; uncited statements are dropped.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import markdown as mdlib

from . import db, llm
from .jobs import JobContext
from .profile import FIELD_LABEL, get_profile, profile_summary_for_prompt
from .textutil import age_days, tokens, truncate

CAT_LABEL = {"customers": "Customers", "competitors": "Competitors", "market": "Market", "internal": "Internal"}
VER_LABEL = {"primary_source": "Verified – primary source", "corroborated": "Reported – corroborated by ≥2 publishers",
             "reported": "Reported – single credible publisher", "user_generated": "User-generated (customer voice, claim unverified)",
             "internal": "Internal data", "unverified": "Unverified"}


def sid(i: dict | int) -> str:
    return f"S{i['id'] if isinstance(i, dict) else i}"


def _interp(it: dict) -> dict:
    return (db.loads(it.get("analysis"), {}) or {}).get("interpretation") or {}


def _period(company_id: int) -> tuple[str, str, str]:
    end = datetime.now(timezone.utc).replace(microsecond=0)
    prev = db.one("SELECT period_end FROM briefs WHERE company_id=? ORDER BY id DESC LIMIT 1", (company_id,))
    if prev and prev["period_end"]:
        start = min(datetime.fromisoformat(prev["period_end"]), end - timedelta(days=7))
        label = f"{start:%d %b} – {end:%d %b %Y}"
    else:
        start = end - timedelta(days=30)
        label = f"{start:%d %b} – {end:%d %b %Y} (first brief: 30-day baseline window)"
    return start.isoformat(), end.isoformat(), label


def _in_window(it: dict, start: str) -> bool:
    if it["published_at"]:
        return it["published_at"] >= start or (it["discovered_at"] >= start and (age_days(it["published_at"]) or 0) <= 45)
    return it["discovered_at"] >= start


SYN_SYSTEM = """You are the chief of staff writing the Monday Morning Intelligence Brief for the founder/CEO of the company below.
Rules:
- Use ONLY the numbered signals provided. Every statement must cite the signal IDs it is based on (e.g. ["S12","S40"]).
- Separate evidence (what the sources say) from your interpretation (what it means for this company).
- Be specific to this company's offerings, segments, geographies and the founder's stated priorities and decisions.
  Generic advice such as "monitor the market" or "improve customer experience" is not acceptable.
- If a category has no meaningful signals, say so; never invent developments, numbers or names.
- Actions must be doable this week, have a clear owner role and a reason tied to cited signals.
- Opportunities and risks must come from what the cited signals show. The founder's priorities may shape the interpretation,
  but never cite a signal for something it does not say. Fewer, well-evidenced items beat padded lists."""

SYN_PROMPT = """Company profile (research + founder interview):
{profile}

Founder's stated decisions this brief should support: {decisions}

Signals this period (ranked; score = combined importance):
{signals}

Category coverage: {coverage}

Return JSON:
{{
 "executive_summary": [{{"headline": "max 18 words", "detail": "1-2 sentences: what happened + why it matters to us", "signal_ids": ["S.."], "polarity": "opportunity|threat|neutral", "urgency": "high|medium|low"}}],
 "customer_summary": "2-3 sentences on customer problems/sentiment/unmet needs, or say there was no meaningful customer signal",
 "competitor_summary": "2-3 sentences",
 "market_summary": "2-3 sentences distinguishing material developments from general news",
 "internal_summary": "1-2 sentences",
 "opportunities": [{{"title": "...", "evidence": "what the signals show", "impact": "expected business impact for us", "next_step": "concrete next step", "priority": "P1|P2|P3", "signal_ids": ["S.."]}}],
 "risks": [{{"title": "...", "impact": "possible impact on us", "urgency": "high|medium|low", "signal_ids": ["S.."]}}],
 "actions": [{{"priority": "P1|P2|P3", "action": "verb-first, specific", "reason": "why, tied to signals", "owner": "role e.g. Founder, Head of Marketing, Partnerships lead", "signal_ids": ["S.."]}}],
 "watch": [{{"topic": "...", "why": "why it deserves continued monitoring", "signal_ids": ["S.."]}}]
}}
Exactly 5 executive_summary items (fewer only if fewer signals exist), at least 3 opportunities, 3-6 risks, 4-7 actions, 3-5 watch items."""


def _signal_line(it: dict, comps: dict) -> str:
    ip = _interp(it)
    comp = f" | competitor={comps.get(it['competitor_id'])}" if it["competitor_id"] else ""
    return (f"{sid(it)} | {it['category']}{comp} | topic={it['topic']} | {it['polarity']} | score={it['score']:.2f} "
            f"rel={it['relevance']:.2f} reliability={it['credibility']:.2f} evidence={it['evidence_strength']:.2f} | "
            f"{it['publisher']} {(it['published_at'] or 'undated')[:10]}\n"
            f"  title: {truncate(it['title'], 160)}\n  evidence: {truncate(it['snippet'] or '(headline only)', 230)}\n"
            + (f"  analyst note: {truncate(ip.get('why_it_matters') or '', 180)}" if ip.get("why_it_matters") else ""))


def generate_brief(company_id: int, trigger: str = "manual", run_id: int | None = None, ctx: JobContext | None = None) -> int:
    step = ctx.step if ctx else (lambda m: None)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    p = get_profile(company_id)
    start, end, label = _period(company_id)
    comps = {c["id"]: c["name"] for c in db.q("SELECT id, name FROM competitors WHERE company_id=?", (company_id,))}
    all_items = db.q("SELECT * FROM items WHERE company_id=? AND dup_of IS NULL ORDER BY score DESC", (company_id,))
    window = [i for i in all_items if _in_window(i, start) and i["score"] is not None]
    sig = [i for i in window if i["significant"]]
    by_cat = {c: [i for i in sig if i["category"] == c] for c in CAT_LABEL}
    n_sources = db.q("SELECT category, COUNT(*) n FROM sources WHERE company_id=? AND status='approved' GROUP BY category", (company_id,))
    src_count = {r["category"]: r["n"] for r in n_sources}
    window_count = Counter(i["category"] for i in window)
    step(f"Brief window {label}: {len(window)} records in window, {len(sig)} significant")

    # ---------------- evidence layer
    top_for_llm: list[dict] = []
    for c in CAT_LABEL:
        top_for_llm += by_cat[c][:10]
    top_for_llm.sort(key=lambda i: -i["score"])
    ids_valid = {sid(i) for i in sig}
    coverage = {CAT_LABEL[c]: f"{len(by_cat[c])} significant of {window_count.get(c, 0)} records from {src_count.get(c, 0)} sources"
                for c in CAT_LABEL}
    decisions = [v.get("text") for v in p.get("intelligence", {}).get("decisions", []) if isinstance(v, dict)]

    data, used = (None, "rule-based")
    if top_for_llm:
        step(f"Synthesising brief with {llm.active_provider_label()}")
        data, used = llm.chat_json(SYN_SYSTEM, SYN_PROMPT.format(
            profile=profile_summary_for_prompt(company_id, p, 2600), decisions="; ".join(decisions) or "(not stated)",
            signals="\n".join(_signal_line(i, comps) for i in top_for_llm[:26]), coverage=json.dumps(coverage)),
            purpose="brief_synthesis", temperature=0.25, max_tokens=3400)
    syn = _validate(data, ids_valid) if isinstance(data, dict) else None
    grounding: dict = {"checked": 0}
    if syn:
        step("Grounding check: verifying every brief statement against its cited evidence")
        grounding = _grounding_check(syn, {sid(i): i for i in sig})
        step(f"Grounding check: {grounding.get('checked', 0)} claims checked, {grounding.get('removed', 0)} removed, "
             f"{grounding.get('partial', 0)} corrected")
        if not syn.get("executive_summary"):
            syn = None
    if not syn:
        syn = _rule_synthesis(company_id, sig, by_cat, comps, p)
        used = "rule-based"
    # Top up from evidence if the model returned too few items (never invent - derive from signals)
    rb = None
    for key, minimum in (("opportunities", 3), ("executive_summary", min(5, len(sig))), ("actions", 3)):
        if len(syn.get(key, [])) < minimum:
            rb = rb or _rule_synthesis(company_id, sig, by_cat, comps, p)
            have = {tuple(x.get("signal_ids", [])) for x in syn.get(key, [])}
            for x in rb.get(key, []):
                if len(syn.setdefault(key, [])) >= minimum:
                    break
                if tuple(x.get("signal_ids", [])) not in have:
                    syn[key].append(x)

    cited = set()
    for key in ("executive_summary", "opportunities", "risks", "actions", "watch"):
        for x in syn.get(key, []):
            cited |= set(x.get("signal_ids", []))
    section_items = {c: by_cat[c][:8] for c in CAT_LABEL}
    for c in CAT_LABEL:
        cited |= {sid(i) for i in section_items[c]}
    customer_block = _customer_block(company_id, window, by_cat["customers"], comps)
    internal_block = _internal_block(company_id, window, by_cat["internal"])
    cited |= {q["id"] for q in customer_block["reviews"]["quotes"]} | {u["id"] for u in customer_block["unmet_needs"]}
    appendix_items = [i for i in window if sid(i) in cited]
    appendix_items.sort(key=lambda i: int(sid(i)[1:]))
    stats = {"sources_approved": sum(src_count.values()), "records_in_window": len(window), "significant": len(sig),
             "by_category": coverage}
    brief = {
        "company": {"id": company_id, "name": company["name"], "website": company["website"]},
        "generated_at": db.now(), "period": {"start": start, "end": end, "label": label}, "llm": used, "stats": stats,
        "grounding": grounding,
        "summaries": {k: syn.get(k) for k in ("customer_summary", "competitor_summary", "market_summary", "internal_summary")},
        "executive_summary": syn.get("executive_summary", [])[:5],
        "customers": {"items": [_card(i, comps) for i in section_items["customers"]], **customer_block},
        "competitors": {"items": [_card(i, comps) for i in section_items["competitors"]]},
        "market": {"items": [_card(i, comps) for i in section_items["market"]]},
        "internal": {"items": [_card(i, comps) for i in section_items["internal"]], **internal_block},
        "opportunities": syn.get("opportunities", []), "risks": syn.get("risks", []), "actions": syn.get("actions", []),
        "watch": syn.get("watch", []),
        "appendix": [_appendix_row(i, comps) for i in appendix_items],
    }
    for c in CAT_LABEL:
        if not brief[c]["items"]:
            brief[c]["none"] = (f"No meaningful {CAT_LABEL[c].lower()} development this period: {window_count.get(c, 0)} records "
                                f"collected from {src_count.get(c, 0)} approved sources, none passed the relevance/significance "
                                f"threshold." if src_count.get(c) else f"No approved {CAT_LABEL[c].lower()} sources yet.")
    md = render_markdown(brief)
    page = render_html(brief, md)
    bid = db.insert("briefs", {"company_id": company_id, "created_at": db.now(), "period_start": start, "period_end": end,
                               "trigger": trigger, "run_id": run_id, "title": f"Monday Brief — {company['name']} — {end[:10]}",
                               "markdown": md, "html": page, "data": brief, "llm": used})
    for a in brief["actions"]:
        db.insert("actions", {"brief_id": bid, "company_id": company_id, "priority": a.get("priority"), "action": a.get("action"),
                              "reason": a.get("reason"), "owner": a.get("owner"), "signal_ids": a.get("signal_ids", []),
                              "status": "open", "updated_at": db.now()})
    step(f"Brief #{bid} saved ({len(brief['appendix'])} cited sources)")
    return bid


GROUND_SYSTEM = """You fact-check an executive intelligence brief before the CEO reads it. Each claim lists the evidence
it cites (verbatim source titles/snippets). Decide whether the claim is grounded in that evidence:
- "yes": every event, number, name and date in the claim appears in the cited evidence; interpretation and recommended
  actions are fine if they follow logically from it.
- "partial": the core is supported but something is added or overstated - give a corrected "fix" that uses only the evidence.
- "no": the cited evidence does not support the claim (wrong citation, invented event, or a claim drawn from somewhere else).
Knowledge about the company itself (its stated priorities) may be used for interpretation, but not as evidence of events."""

GROUND_PROMPT = """Claims to check:
{claims}

Return JSON: {{"checks": [{{"id": "<claim id>", "supported": "yes|partial|no", "fix": "corrected claim text if partial, else empty", "note": "short reason"}}]}}"""

_CLAIM_TEXT = {
    "executive_summary": lambda x: f"{x.get('headline', '')} — {x.get('detail', '')}",
    "opportunities": lambda x: f"{x.get('title', '')}. Evidence: {x.get('evidence', '')}. Impact: {x.get('impact', '')}",
    "risks": lambda x: f"{x.get('title', '')}. Impact: {x.get('impact', '')}",
    "actions": lambda x: f"{x.get('action', '')} — because {x.get('reason', '')}",
    "watch": lambda x: f"{x.get('topic', '')} — {x.get('why', '')}",
}
_FIX_FIELD = {"executive_summary": "detail", "opportunities": "evidence", "risks": "impact", "actions": "reason", "watch": "why"}


def _grounding_check(syn: dict, by_sid: dict[str, dict]) -> dict:
    """Second-pass verification: drop brief statements whose cited signals do not support them."""
    claims, index = [], {}
    for key in _CLAIM_TEXT:
        for n, x in enumerate(syn.get(key, [])):
            cid = f"{key[:3].upper()}{n + 1}"
            index[cid] = (key, x)
            ev = "\n".join(f"    {s}: {truncate(by_sid[s]['title'], 140)} — {truncate(by_sid[s]['snippet'] or '', 260)}"
                           for s in x.get("signal_ids", []) if s in by_sid)
            claims.append(f"[{cid}] {truncate(_CLAIM_TEXT[key](x), 420)}\n  cited evidence:\n{ev}")
    if not claims:
        return {"checked": 0}
    checks: dict[str, dict] = {}
    used = None
    for b in range(0, len(claims), 12):
        data, used = llm.chat_json(GROUND_SYSTEM, GROUND_PROMPT.format(claims="\n\n".join(claims[b:b + 12])),
                                   purpose="brief_grounding", temperature=0.0, max_tokens=2200)
        for c in (data or {}).get("checks", []) if isinstance(data, dict) else []:
            if isinstance(c, dict) and c.get("id") in index:
                checks[c["id"]] = c
    if not checks:
        return {"checked": 0, "note": "grounding check unavailable (no model answered)"}
    removed, partial = [], 0
    keep: dict[str, list] = {k: [] for k in _CLAIM_TEXT}
    for cid, (key, x) in index.items():
        c = checks.get(cid) or {}
        verdict = str(c.get("supported", "yes")).lower()
        if verdict == "no":
            removed.append({"section": key, "claim": truncate(_CLAIM_TEXT[key](x), 160), "reason": c.get("note", "")})
            continue
        if verdict == "partial":
            partial += 1
            if c.get("fix"):
                x[_FIX_FIELD[key]] = c["fix"]
            x["grounding"] = "partially supported — corrected to the evidence"
        keep[key].append(x)
    for key in _CLAIM_TEXT:
        syn[key] = keep[key]
    return {"checked": len(index), "removed": len(removed), "partial": partial, "removed_claims": removed, "by": used}


def _validate(data: dict, ids_valid: set[str]) -> dict | None:
    out = {}
    for k in ("customer_summary", "competitor_summary", "market_summary", "internal_summary"):
        out[k] = str(data.get(k) or "")
    for key in ("executive_summary", "opportunities", "risks", "actions", "watch"):
        rows = []
        for x in data.get(key) or []:
            if not isinstance(x, dict):
                continue
            ids = [s for s in (x.get("signal_ids") or []) if isinstance(s, str) and s.strip() in ids_valid]
            if not ids:
                continue  # uncited claims are dropped
            x["signal_ids"] = list(dict.fromkeys(s.strip() for s in ids))
            rows.append(x)
        out[key] = rows
    return out if out["executive_summary"] else None


def _rule_synthesis(company_id: int, sig: list[dict], by_cat: dict, comps: dict, p: dict) -> dict:
    def why(i):
        return _interp(i).get("why_it_matters") or i["explanation"]

    def consider(i):
        return _interp(i).get("consider") or "Review with the team."

    from .analysis import OWNERS
    top = sorted(sig, key=lambda i: -i["score"])
    ex = [{"headline": truncate(i["title"], 110), "detail": f"{why(i)}", "signal_ids": [sid(i)],
           "polarity": i["polarity"], "urgency": "high" if i["urgency"] > 0.7 else "medium" if i["urgency"] > 0.45 else "low"}
          for i in top[:5]]
    opps = [i for i in top if i["polarity"] == "opportunity"][:5]
    threats = [i for i in top if i["polarity"] == "threat"][:5]
    return {
        "executive_summary": ex,
        "customer_summary": "", "competitor_summary": "", "market_summary": "", "internal_summary": "",
        "opportunities": [{"title": truncate(i["title"], 100), "evidence": truncate(i["snippet"] or i["title"], 220),
                           "impact": why(i), "next_step": consider(i), "priority": "P1" if i["score"] > 0.6 else "P2",
                           "signal_ids": [sid(i)]} for i in opps],
        "risks": [{"title": truncate(i["title"], 100), "impact": why(i),
                   "urgency": "high" if i["urgency"] > 0.7 else "medium" if i["urgency"] > 0.45 else "low",
                   "signal_ids": [sid(i)]} for i in threats],
        "actions": [{"priority": "P1" if n < 2 else "P2", "action": consider(i), "reason": f"{truncate(i['title'], 90)} — {why(i)}",
                     "owner": OWNERS.get(i["topic"] or "other", "Founder"), "signal_ids": [sid(i)]} for n, i in enumerate(top[:6])],
        "watch": [{"topic": truncate(i["title"], 90), "why": f"Early signal (novelty {i['novelty']:.2f}); not yet decision-relevant.",
                   "signal_ids": [sid(i)]} for i in sorted(sig, key=lambda i: -(i["novelty"] or 0))[5:9]],
    }


def _card(i: dict, comps: dict) -> dict:
    ip = _interp(i)
    return {"id": sid(i), "title": i["title"], "url": i["url"], "publisher": i["publisher"],
            "published_at": (i["published_at"] or "")[:10] or None, "discovered_at": i["discovered_at"][:10],
            "competitor": comps.get(i["competitor_id"]), "topic": i["topic"], "polarity": i["polarity"],
            "evidence": truncate(i["snippet"] or "", 400), "what_happened": ip.get("what_happened"),
            "why_it_matters": ip.get("why_it_matters"), "consider": ip.get("consider"), "business_area": ip.get("business_area"),
            "verification": i["verification"], "reliability": i["credibility"], "evidence_strength": i["evidence_strength"],
            "relevance": i["relevance"], "score": i["score"], "explanation": i["explanation"],
            "interpreted_by": (db.loads(i.get("analysis"), {}) or {}).get("interpreted_by")}


def _appendix_row(i: dict, comps: dict) -> dict:
    return {"id": sid(i), "source": i["publisher"], "title": i["title"], "url": i["url"],
            "published_at": (i["published_at"] or "")[:10] or "undated", "discovered_at": i["discovered_at"][:10],
            "snippet": truncate(i["snippet"] or i["title"], 260), "verification": VER_LABEL.get(i["verification"], i["verification"]),
            "reliability": i["credibility"], "evidence_strength": i["evidence_strength"], "relevance": i["relevance"]}


def _customer_block(company_id: int, window: list[dict], sig_customers: list[dict], comps: dict) -> dict:
    reviews = [i for i in window if i["category"] == "customers" and (i["publisher"] or "").startswith("App Store review")]
    ratings = []
    for r in reviews:
        extra = (db.loads(r.get("analysis"), {}) or {}).get("extra") or {}
        if extra.get("rating"):
            ratings.append(extra["rating"])
        else:
            m = re.match(r"★(\d)", r["title"])
            if m:
                ratings.append(int(m.group(1)))
    neg = [r for r in reviews if (r["sentiment"] or 0) < -0.05 or r["title"].startswith(("★1", "★2"))]
    c = Counter()
    for r in neg:
        toks = tokens(r["snippet"] or "")
        c.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    themes = [t for t, n in c.most_common(8) if n >= 2]
    quotes = [{"quote": truncate(r["snippet"], 220), "url": r["url"], "source": r["publisher"], "id": sid(r)}
              for r in sorted(neg, key=lambda r: r["sentiment"] or 0)[:4]]
    asks = [r for r in window if r["category"] == "customers" and re.search(r"\b(wish|please add|should have|would love|need)\b",
                                                                              r["snippet"] or "", re.I)]
    sents = [i["sentiment"] for i in window if i["category"] == "customers" and i["sentiment"] is not None]
    return {"reviews": {"count": len(reviews), "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                        "negative": len(neg), "complaint_phrases": themes, "quotes": quotes},
            "unmet_needs": [{"text": truncate(r["snippet"], 200), "url": r["url"], "id": sid(r)} for r in asks[:4]],
            "sentiment": {"mean": round(sum(sents) / len(sents), 3) if sents else None, "n": len(sents)}}


def _internal_block(company_id: int, window: list[dict], sig_internal: list[dict]) -> dict:
    ups = db.q("SELECT * FROM uploads WHERE company_id=? ORDER BY id DESC LIMIT 5", (company_id,))
    themes, risks, synthetic = [], [], False
    for u in ups:
        s = db.loads(u["summary"], {}) or {}
        themes += [{**t, "file": u["filename"]} for t in s.get("themes", [])]
        risks += s.get("emerging_risks", [])
        synthetic = synthetic or bool(u["is_synthetic"])
    labels = Counter(i["topic"] for i in window if i["category"] == "internal")
    return {"themes": themes[:8], "emerging_risks": risks[:5], "label_counts": dict(labels),
            "uploads": [{"filename": u["filename"], "records": u["records"], "synthetic": bool(u["is_synthetic"])} for u in ups],
            "synthetic_notice": synthetic}


# --------------------------------------------------------------------------- rendering
def _ids(x: dict) -> str:
    return " ".join(f"[{s}](#{s})" for s in x.get("signal_ids", []))


def render_markdown(b: dict) -> str:
    L: list[str] = []
    A = L.append
    A(f"# Monday Morning Intelligence Brief — {b['company']['name']}")
    A(f"*Period: {b['period']['label']} · generated {b['generated_at'][:16].replace('T', ' ')} UTC · "
      f"analysis: {b['llm']} · {b['stats']['sources_approved']} approved sources · {b['stats']['records_in_window']} records in window · "
      f"{b['stats']['significant']} significant*")
    A("")
    A("> **How to read this brief.** *Evidence* (titles, snippets, dates, links) comes verbatim from the cited sources. "
      "*Interpretation* (why it matters, what to do) is the system's analysis and is labelled as such. Source reliability, "
      "evidence strength and strategic relevance are scored separately; the AI's own confidence is never used as proof. "
      "Signal IDs (S…) link to the Source Appendix.")
    g = b.get("grounding") or {}
    if g.get("checked"):
        A(f">\n> **Grounding check:** a second model pass compared {g['checked']} statements with the evidence they cite — "
          f"{g.get('removed', 0)} unsupported statement(s) removed, {g.get('partial', 0)} corrected to match the evidence.")
    A("")
    A("## 1. Executive Summary")
    if not b["executive_summary"]:
        A("No significant developments were detected in this period.")
    for n, x in enumerate(b["executive_summary"], 1):
        A(f"{n}. **{x.get('headline')}** — {x.get('detail')} "
          f"*({x.get('polarity', 'neutral')}, urgency {x.get('urgency', '—')})* {_ids(x)}")
    A("")

    def section(num, title, key, summary_key):
        A(f"## {num}. {title}")
        s = (b.get("summaries") or {}).get(summary_key)
        if s:
            A(f"*Interpretation:* {s}")
            A("")
        if b[key].get("none"):
            A(f"**{b[key]['none']}**")
            A("")
        for c in b[key]["items"]:
            comp = f" · competitor: **{c['competitor']}**" if c.get("competitor") else ""
            A(f"- <a id='{c['id']}-card'></a>**{c['title']}** — {c['publisher']}, {c['published_at'] or 'undated'}{comp} "
              f"· [{c['id']}](#{c['id']}) · [link]({c['url']})")
            if c.get("evidence"):
                A(f"  - *Evidence:* “{c['evidence']}”")
            if c.get("why_it_matters"):
                A(f"  - *Why it matters (interpretation):* {c['why_it_matters']}")
            if c.get("consider"):
                A(f"  - *Consider:* {c['consider']}")
            A(f"  - *Scores:* relevance {c['relevance']:.2f} · source reliability {c['reliability']:.2f} · evidence strength "
              f"{c['evidence_strength']:.2f} · {VER_LABEL.get(c['verification'], c['verification'])} · {c['polarity']}")
        A("")

    section(2, "Customer Signals", "customers", "customer_summary")
    cb = b["customers"]
    if cb["reviews"]["count"]:
        r = cb["reviews"]
        A(f"**Customer voice (app reviews in window):** {r['count']} reviews"
          + (f", average rating {r['avg_rating']}" if r["avg_rating"] else "") + f", {r['negative']} negative."
          + (f" Recurring complaint phrases: {', '.join(r['complaint_phrases'])}." if r["complaint_phrases"] else ""))
        for q in r["quotes"]:
            A(f"> “{q['quote']}” — {q['source']} [{q['id']}](#{q['id']})")
        A("")
    if cb["unmet_needs"]:
        A("**Unmet needs expressed by customers:**")
        for u in cb["unmet_needs"]:
            A(f"- “{u['text']}” [{u['id']}](#{u['id']})")
        A("")
    if cb["sentiment"]["n"]:
        A(f"*Average sentiment of customer records (VADER, −1…+1): {cb['sentiment']['mean']} over {cb['sentiment']['n']} records.*")
        A("")
    section(3, "Competitor Intelligence", "competitors", "competitor_summary")
    section(4, "Market Intelligence", "market", "market_summary")
    section(5, "Internal Business Intelligence", "internal", "internal_summary")
    ib = b["internal"]
    if ib.get("synthetic_notice"):
        A("> ⚠️ **Internal data in this brief comes from a SYNTHETIC TEST FILE** uploaded to demonstrate the pipeline. "
          "It is not real company data.")
    if ib["themes"]:
        A("**Recurring themes in uploaded internal data:**")
        for t in ib["themes"]:
            A(f"- **{t.get('theme')}** ({t.get('label')}, {t.get('count')} records, {t.get('file')}): {t.get('insight')}"
              + (f" — e.g. “{truncate(str(t.get('example')), 140)}”" if t.get("example") else ""))
    if ib["emerging_risks"]:
        A("**Emerging internal risks:** " + "; ".join(map(str, ib["emerging_risks"])))
    if not ib["uploads"]:
        A("No internal data has been uploaded yet (CSV, Excel, PDF and text are supported in the Source Manager).")
    A("")
    A("## 6. Opportunities Identified")
    if not b["opportunities"]:
        A("No evidence-backed opportunities this period.")
    for n, o in enumerate(b["opportunities"], 1):
        A(f"{n}. **{o.get('title')}** — priority **{o.get('priority', 'P2')}** {_ids(o)}")
        A(f"   - *Evidence:* {o.get('evidence', '')}")
        A(f"   - *Impact (interpretation):* {o.get('impact', '')}")
        A(f"   - *Next step:* {o.get('next_step', '')}")
    A("")
    A("## 7. Risks & Threats")
    if not b["risks"]:
        A("No material negative developments detected this period.")
    for n, r in enumerate(b["risks"], 1):
        A(f"{n}. **{r.get('title')}** — urgency **{r.get('urgency', 'medium')}** {_ids(r)}")
        A(f"   - *Possible impact:* {r.get('impact', '')}")
    A("")
    A("## 8. Recommended Actions for This Week")
    if b["actions"]:
        A("| Priority | Action | Reason | Suggested owner | Signals |")
        A("|---|---|---|---|---|")
        for a in b["actions"]:
            A(f"| {a.get('priority')} | {a.get('action')} | {str(a.get('reason', '')).replace('|', '/')} | {a.get('owner')} | {_ids(a)} |")
    else:
        A("No actions recommended this week.")
    A("")
    A("## 9. What We Should Watch")
    if not b["watch"]:
        A("Nothing new to add to the watch-list.")
    for w in b["watch"]:
        A(f"- **{w.get('topic')}** — {w.get('why')} {_ids(w)}")
    A("")
    A("## 10. Source Appendix")
    A("| ID | Source | Title / link | Published | Discovered | Evidence snippet | Verification | Reliability | Evidence | Relevance |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in b["appendix"]:
        snip = str(r["snippet"]).replace("|", "/").replace("\n", " ")
        title = str(r["title"]).replace("|", "/").replace("[", "(").replace("]", ")")
        A(f"| <a id='{r['id']}'></a>{r['id']} | {r['source']} | [{truncate(title, 90)}]({r['url']}) | {r['published_at']} | "
          f"{r['discovered_at']} | {truncate(snip, 200)} | {r['verification']} | {r['reliability']:.2f} | "
          f"{r['evidence_strength']:.2f} | {r['relevance']:.2f} |")
    A("")
    A("*Generated by Signal Stack. Facts are traceable to the links above; anything not in a cited source is interpretation.*")
    return "\n".join(L)


BRIEF_CSS = """
body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#1b1f24;line-height:1.55}
h1{font-size:26px;margin-bottom:4px}h2{font-size:19px;margin-top:32px;border-bottom:2px solid #e6e8eb;padding-bottom:6px;color:#0f3d6e}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}th,td{border:1px solid #dde1e6;padding:6px 7px;vertical-align:top;text-align:left}
th{background:#f3f5f8}blockquote{border-left:4px solid #9db7d5;margin:10px 0;padding:6px 14px;background:#f6f9fc;color:#33414f}
a{color:#0b62c4}code{background:#f2f2f2;padding:1px 4px}li{margin:4px 0}em{color:#45505c}
@media print{body{margin:0;max-width:none}a{color:#000}}
"""


def render_html(b: dict, md: str | None = None) -> str:
    md = md or render_markdown(b)
    body = mdlib.markdown(md, extensions=["tables"])
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Monday Brief — {html.escape(b['company']['name'])}</title>"
            f"<style>{BRIEF_CSS}</style></head><body>{body}</body></html>")


def email_html(brief_row: dict) -> str:
    """Email-ready version: inline styles only (email clients strip <style>)."""
    body = mdlib.markdown(brief_row["markdown"], extensions=["tables"])
    body = (body.replace("<table>", "<table style='border-collapse:collapse;width:100%;font-size:12px'>")
            .replace("<th>", "<th style='border:1px solid #ccc;padding:5px;background:#f3f5f8;text-align:left'>")
            .replace("<td>", "<td style='border:1px solid #ccc;padding:5px;vertical-align:top'>")
            .replace("<h2>", "<h2 style='color:#0f3d6e;font-size:18px;border-bottom:2px solid #e6e8eb'>")
            .replace("<blockquote>", "<blockquote style='border-left:4px solid #9db7d5;margin:8px 0;padding:4px 12px;background:#f6f9fc'>"))
    return (f"<div style='font-family:Segoe UI,Arial,sans-serif;max-width:900px;color:#1b1f24;line-height:1.5'>{body}</div>")
