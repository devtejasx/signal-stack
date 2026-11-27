"""Business-context change: update the profile, reassess every source against the new
context, retire sources that no longer fit, discover new ones and explain what changed."""
from __future__ import annotations

from . import db, llm
from .discovery import build_signal_stack
from .jobs import JobContext
from .profile import get_profile, interest_document, profile_summary_for_prompt, save_profile
from .textutil import clean_ws, key_phrases, sim_to_reference, truncate

UPDATE_SYSTEM = """You maintain a company's intelligence profile. The founder has told you how the business context changed.
Update the search/monitoring fields so they reflect the NEW context. Drop topics that no longer matter."""
UPDATE_PROMPT = """Current profile:
{profile}

Current monitoring fields:
category_keywords: {kw}
customer_problem_queries: {cq}
market_topics: {mt}
regulatory_topics: {rt}
geographies: {geo}

Context change from the founder:
\"\"\"{change}\"\"\"

Return JSON with the updated lists and a short explanation:
{{"category_keywords": [], "customer_problem_queries": [], "market_topics": [], "regulatory_topics": [], "geographies": [],
  "competitor_candidates": [{{"name": "", "domain": "", "kind": "direct|substitute", "why": ""}}],
  "dropped": ["topics no longer relevant"], "added": ["new topics"], "explanation": "2-3 sentences"}}"""

LIST_FIELDS = ("category_keywords", "customer_problem_queries", "market_topics", "regulatory_topics", "geographies")


def apply_context_change(company_id: int, change: str, ctx: JobContext | None = None) -> int:
    step = ctx.step if ctx else (lambda m: None)
    p_old = get_profile(company_id)
    old_doc = interest_document(company_id, p_old)
    step("Updating the company profile with the new business context")
    data, used = llm.chat_json(UPDATE_SYSTEM, UPDATE_PROMPT.format(
        profile=profile_summary_for_prompt(company_id, p_old, 2500),
        kw=p_old.get("category_keywords"), cq=p_old.get("customer_problem_queries"), mt=p_old.get("market_topics"),
        rt=p_old.get("regulatory_topics"), geo=p_old.get("geographies"), change=change), purpose="context_change", temperature=0.2)
    p_new = dict(p_old)
    p_new["intelligence"] = dict(p_old.get("intelligence") or {})
    p_new["intelligence"]["growth_priorities"] = [*p_new["intelligence"].get("growth_priorities", []),
                                                  {"text": change, "source": "context_change"}]
    explanation = ""
    if isinstance(data, dict):
        for k in LIST_FIELDS:
            if isinstance(data.get(k), list) and data[k]:
                p_new[k] = [clean_ws(str(x)) for x in data[k] if str(x).strip()][:12]
        if isinstance(data.get("competitor_candidates"), list) and data["competitor_candidates"]:
            p_new["competitor_candidates"] = [c for c in data["competitor_candidates"] if isinstance(c, dict) and c.get("name")]
        explanation = clean_ws(str(data.get("explanation") or ""))
    else:
        phrases = key_phrases([change], 8)
        p_new["category_keywords"] = list(dict.fromkeys(phrases[:4] + p_old.get("category_keywords", [])))[:10]
        p_new["market_topics"] = list(dict.fromkeys([f"{x} market" for x in phrases[:3]] + p_old.get("market_topics", [])))[:8]
    p_new.setdefault("context_changes", []).append({"at": db.now(), "change": change})
    save_profile(company_id, p_new, f"business context change: {truncate(change, 80)}")
    new_doc = interest_document(company_id, p_new)

    # ---------------------------------------------------------------- reassess every source
    step("Reassessing relevance of every approved source against the new context")
    sources = db.q("SELECT * FROM sources WHERE company_id=? AND status='approved' AND type!='upload'", (company_id,))
    retired, kept = [], []
    for s in sources:
        items = db.q("SELECT title, snippet FROM items WHERE source_id=? ORDER BY id DESC LIMIT 25", (s["id"],))
        texts = [f"{i['title']} {i['snippet'] or ''}" for i in items] or [f"{s['name']} {s['query'] or ''} {s['reason']}"]
        def_text = f"{s['name']} {s['query'] or ''}"
        old_r = sum(sim_to_reference(old_doc, texts)) / len(texts)
        new_r = sum(sim_to_reference(new_doc, texts)) / len(texts)
        def_new = sim_to_reference(new_doc, [def_text])[0]
        row = {"id": s["id"], "name": s["name"], "category": s["category"], "old": round(old_r, 3), "new": round(new_r, 3)}
        # Competitor sources depend on the competitor still being relevant; topic sources on their query.
        drop_topic = s["query"] and s["category"] == "market" and not any(
            t.lower() in (s["query"] or "").lower() or (s["query"] or "").lower() in t.lower()
            for t in p_new.get("market_topics", []) + p_new.get("regulatory_topics", []) + p_new.get("category_keywords", []))
        drop_cust = s["query"] and s["category"] == "customers" and s["type"] == "bing_news" and \
            s["query"] not in p_new.get("customer_problem_queries", [])
        if (drop_topic or drop_cust) or (new_r < 0.6 * old_r and def_new < 0.05):
            row["why"] = ("its topic is no longer in the updated monitoring plan" if (drop_topic or drop_cust)
                          else f"relevance to the new context fell from {old_r:.2f} to {new_r:.2f}")
            db.update("sources", s["id"], {"status": "removed", "removed_at": db.now(),
                                           "reason": s["reason"] + f" [Retired after context change: {row['why']}]"})
            retired.append(row)
        else:
            kept.append(row)

    before = {s["id"] for s in db.q("SELECT id FROM sources WHERE company_id=?", (company_id,))}
    step("Discovering sources for the new context")
    build_signal_stack(company_id, ctx, origin="context_change")
    added = [{"id": s["id"], "name": s["name"], "category": s["category"], "status": s["status"], "reason": s["reason"]}
             for s in db.q("SELECT * FROM sources WHERE company_id=?", (company_id,)) if s["id"] not in before]
    from .analysis import score_items
    score_items(company_id, ctx=ctx)

    diff = {"change": change, "explanation": explanation, "by": used, "dropped_topics": (data or {}).get("dropped", []) if isinstance(data, dict) else [],
            "added_topics": (data or {}).get("added", []) if isinstance(data, dict) else [],
            "fields_before": {k: p_old.get(k) for k in LIST_FIELDS}, "fields_after": {k: p_new.get(k) for k in LIST_FIELDS},
            "retired_sources": retired, "kept_sources": kept, "added_sources": added}
    md = _markdown(diff)
    rid = db.insert("stack_revisions", {"company_id": company_id, "created_at": db.now(), "reason": change, "diff": diff, "markdown": md})
    step(f"Context change applied: {len(retired)} sources retired, {len(added)} added, {len(kept)} kept")
    return rid


def _markdown(d: dict) -> str:
    L = ["# Signal Stack revision after business-context change", "", f"**What changed:** {d['change']}", ""]
    if d["explanation"]:
        L += [f"**System assessment:** {d['explanation']}", ""]
    if d["dropped_topics"]:
        L.append("**Topics dropped:** " + "; ".join(map(str, d["dropped_topics"])))
    if d["added_topics"]:
        L.append("**Topics added:** " + "; ".join(map(str, d["added_topics"])))
    L += ["", "## Monitoring plan: before → after"]
    for k in LIST_FIELDS:
        L.append(f"- **{k}**: {', '.join(d['fields_before'].get(k) or [])} → **{', '.join(d['fields_after'].get(k) or [])}**")
    L += ["", f"## Sources retired ({len(d['retired_sources'])})"]
    L += [f"- {r['name']} ({r['category']}) — {r['why']}" for r in d["retired_sources"]] or ["None."]
    L += ["", f"## Sources added ({len(d['added_sources'])})"]
    L += [f"- {a['name']} ({a['category']}, {a['status']}) — {truncate(a['reason'], 220)}" for a in d["added_sources"]] or ["None."]
    L += ["", f"## Sources kept ({len(d['kept_sources'])})"]
    L += [f"- {k['name']} (relevance {k['old']:.2f} → {k['new']:.2f})" for k in d["kept_sources"][:40]]
    return "\n".join(L)
