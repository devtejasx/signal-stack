"""Adaptive founder interview.

* 10 core questions are planned from the research (one per intelligence area), worded
  around what research found / could not confirm about *this* company.
* Asked one at a time. After every answer the system
    1. extracts structured statements into the intelligence profile,
    2. decides whether a follow-up is warranted (max one per question, max 4 in total),
    3. re-words the next planned question so it builds on what was said and never repeats.
* Questions can be skipped (confidential) and answers edited later.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from . import db, llm
from .profile import FIELD_LABEL, INTEL_FIELDS, get_profile, profile_summary_for_prompt, save_profile
from .textutil import clean_ws, truncate

MAX_FOLLOWUPS = 4
AREAS = [k for k, _ in INTEL_FIELDS]

PLAN_SYSTEM = """You are an experienced strategy consultant preparing a founder interview. You receive 10 draft
questions that were written from research on this company. Rewrite each draft so it is sharp, natural and
specific to THIS company. Keep every concrete detail in the draft (offerings, segments, places, claims, topics).
Do not add company or competitor names that are not in the draft or the profile. One question per area."""

PLAN_PROMPT = """Research profile:
{profile}

Unconfirmed assumptions the interview should confirm or correct:
{assumptions}

Draft questions (area: draft):
{drafts}

Return JSON: {{"questions": [{{"area": "<area key>", "question": "rewritten question, max 45 words"}}]}}"""

FOLLOW_SYSTEM = """You are conducting a founder interview for a business-intelligence system.
Extract what the founder said into short structured statements and decide if ONE follow-up question would
materially improve the intelligence profile (e.g. the answer named something worth monitoring, was vague,
or contradicted the research). Do not ask a follow-up for complete, specific answers."""

FOLLOW_PROMPT = """Company context:
{profile}

Interview so far:
{history}

Current question ({area}): {question}
Founder answer: {answer}

Return JSON:
{{"extracted": ["3-6 short statements capturing the answer, in the founder's terms"],
  "entities": {{"competitors": [], "segments": [], "geographies": [], "topics": []}},
  "followup": "a single follow-up question, or empty string if not needed",
  "followup_reason": "why"}}"""

REFINE_SYSTEM = """You adapt the next planned interview question so that it builds on the founder's previous answers.
Keep the same intent/area, reference relevant earlier answers when useful, never repeat something already
answered, max 45 words."""

REFINE_PROMPT = """Company: {name}
Previous Q&A:
{history}

Next planned question ({area}): {question}

Return JSON: {{"question": "the adapted question"}}"""


# --------------------------------------------------------------------------- planning
def _rule_plan(company_id: int) -> list[dict]:
    """Grounded draft questions built from this company's own research (no unverified names)."""
    p = get_profile(company_id)
    name = p.get("name", "the company")

    def first(key: str, n: int, default: str) -> str:
        vals = [truncate(re.sub(r"\s*\(.*?\)", "", str(v)).strip(), 45) for v in (p.get(key) or []) if str(v).strip()][:n]
        return ", ".join(vals) if vals else default

    off_s = first("offerings", 3, "your main offerings")
    seg_s = first("customer_segments", 2, "the customers your website targets")
    geo_s = first("geographies", 4, "your key markets")
    kw = p.get("category_keywords") or []
    mt = p.get("market_topics") or []
    rt = p.get("regulatory_topics") or []
    vp = p.get("value_props") or []
    bm = p.get("business_model") or ""
    industry = p.get("industry") or "your category"
    assumption = db.one("SELECT statement FROM facts WHERE company_id=? AND kind='assumption' AND field='inference' LIMIT 1",
                        (company_id,))
    drafts = {
        "business_lines": (f"Your website presents {off_s}. Which of these actually drives most of {name}'s revenue today, "
                           f"and how do you charge for it" + (f" — is it really {bm.lower().rstrip('.')}?" if bm and 'unknown' not in bm else "?"),
                           f"Research found these offerings on the site; the revenue model ('{bm or 'unknown'}') is an unconfirmed assumption."),
        "priority_segments": (f"Research suggests you serve {seg_s}. Which segment is the most valuable to {name}, and is it growing?",
                              "Segments were inferred from site copy, not confirmed."),
        "customer_problems": (f"Before customers find {name}, what goes wrong for them with " +
                              (f"{kw[0]} and {kw[1]}" if len(kw) > 1 else (kw[0] if kw else industry)) +
                              "? What do they say, in their own words?",
                              "Needed to choose where customer conversations should be monitored."),
        "differentiators": ((f"Your site claims “{truncate(vp[0], 90)}”. " if vp else "") +
                            f"When a customer picks {name} over alternatives, what is the real deciding reason?",
                            "Positioning claim taken from the website; the founder confirms what actually wins customers."),
        "competitors": (f"Who do you most often lose customers to — other {industry} players, cheaper local options, or substitutes? "
                        "Please name them.", "Competitor candidates from research are verified separately; the founder's view comes first."),
        "market_changes": (f"Which changes in {', '.join(mt[:2]) if mt else industry} are changing how your customers in {geo_s} buy right now?",
                           "Market topics proposed by research, to be confirmed or replaced."),
        "material_risks": (f"Which external development" + (f" — for example around {rt[0].lower()} —" if rt else "") +
                           f" could hurt {name} most in the next 12 months?", "Prioritises regulatory and risk sources."),
        "growth_priorities": (f"What are {name}'s top growth priorities for the next 6-12 months: new geographies beyond {geo_s}, "
                              f"new segments, or new offerings?", "Growth priorities decide which opportunities the brief should surface."),
        "internal_data": ("Which internal data do you trust most — CRM exports, sales-call notes, support tickets or reviews — "
                          "and what is it telling you right now?", "Configures the Internal intelligence category."),
        "decisions": (f"Which recurring decisions should the Monday brief help you make at {name} — for example pricing, which markets "
                      "to prioritise, or which partnerships to pursue?", "Defines what 'decision-relevant' means for this founder."),
    }
    if assumption:
        drafts["business_lines"] = (drafts["business_lines"][0], drafts["business_lines"][1] + f" Assumption: {truncate(assumption['statement'], 120)}")
    return [{"area": a, "question": drafts[a][0], "rationale": drafts[a][1]} for a in AREAS]


def plan_interview(company_id: int) -> list[dict]:
    existing = db.q("SELECT * FROM interview_questions WHERE company_id=? ORDER BY seq, is_followup, id", (company_id,))
    if existing:
        return existing
    p = get_profile(company_id)
    drafts = _rule_plan(company_id)
    assumptions = db.q("SELECT statement FROM facts WHERE company_id=? AND kind='assumption' LIMIT 8", (company_id,))
    data, used = llm.chat_json(PLAN_SYSTEM, PLAN_PROMPT.format(
        profile=profile_summary_for_prompt(company_id, p, 2000),
        assumptions="\n".join("- " + truncate(a["statement"], 200) for a in assumptions) or "- (none)",
        drafts="\n".join(f"- {d['area']}: {d['question']}" for d in drafts)), purpose="interview_plan", temperature=0.3)
    polished = {}
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        for qd in data["questions"]:
            if isinstance(qd, dict) and str(qd.get("area", "")).strip() in AREAS and qd.get("question"):
                polished.setdefault(str(qd["area"]).strip(), clean_ws(str(qd["question"])))
    plan = []
    for d in drafts:
        q = polished.get(d["area"])
        # Keep the rewrite only if it preserved the draft's specifics (small models tend to genericise).
        if q and 15 < len(q) < 400 and fuzz.token_set_ratio(d["question"], q) >= 55:
            plan.append({**d, "question": q, "generated_by": f"{used} (polished research draft)"})
        else:
            plan.append({**d, "generated_by": "research draft"})
    for i, qd in enumerate(plan, start=1):
        db.insert("interview_questions", {"company_id": company_id, "seq": i, "area": qd["area"], "question": qd["question"],
                                          "rationale": qd.get("rationale"), "status": "pending",
                                          "generated_by": qd.get("generated_by"), "updated_at": db.now()})
    db.update("companies", company_id, {"status": "interviewing", "updated_at": db.now()})
    return db.q("SELECT * FROM interview_questions WHERE company_id=? ORDER BY seq, is_followup, id", (company_id,))


# --------------------------------------------------------------------------- flow
def ordered(company_id: int) -> list[dict]:
    return db.q("SELECT * FROM interview_questions WHERE company_id=? ORDER BY seq, is_followup, id", (company_id,))


def next_question(company_id: int) -> dict | None:
    for qrow in ordered(company_id):
        if qrow["status"] == "pending":
            if not qrow["asked_at"]:
                db.update("interview_questions", qrow["id"], {"asked_at": db.now()})
            return qrow
    return None


def progress(company_id: int) -> dict:
    rows = ordered(company_id)
    core = [r for r in rows if not r["is_followup"]]
    done = [r for r in core if r["status"] != "pending"]
    return {"core_total": len(core), "core_done": len(done), "followups": sum(r["is_followup"] for r in rows),
            "complete": bool(core) and len(done) == len(core) and all(r["status"] != "pending" for r in rows)}


def _history(company_id: int, limit: int = 12) -> str:
    rows = [r for r in ordered(company_id) if r["status"] != "pending"][-limit:]
    out = []
    for r in rows:
        ans = r["answer"] if r["status"] == "answered" else "(skipped - confidential)"
        out.append(f"Q ({r['area']}): {r['question']}\nA: {truncate(ans, 600)}")
    return "\n\n".join(out) or "(none yet)"


def _rule_extract(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+|\s*•\s*", answer)
    return [clean_ws(p) for p in parts if len(clean_ws(p)) > 3][:8]


def _apply_to_profile(company_id: int, qrow: dict, statements: list[str], entities: dict | None, reason: str) -> None:
    p = get_profile(company_id)
    intel = p.setdefault("intelligence", {})
    area = qrow["area"]
    kept = [v for v in intel.get(area, []) if not (isinstance(v, dict) and v.get("question_id") == qrow["id"])]
    kept += [{"text": s, "question_id": qrow["id"], "source": "interview"} for s in statements]
    intel[area] = kept
    if area == "competitors" and not (entities or {}).get("competitors"):
        # Fallback when the model returns no entities: capitalised names in the founder's own words.
        text = " ".join(statements)
        names = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s(?:[A-Z][a-zA-Z0-9]+|Edu|Abroad|Scholar))*)\b", text)
        stop = {"Large", "We", "Our", "The", "Watch", "Indian", "India", "Germany", "UK", "USA", "Canada", "Fintech"}
        entities = {**(entities or {}), "competitors": [n for n in dict.fromkeys(names) if n not in stop and len(n) > 2]}
    if entities:
        named = p.setdefault("founder_named", {})
        for k in ("competitors", "segments", "geographies", "topics"):
            vals = [clean_ws(str(v)) for v in entities.get(k, []) or [] if str(v).strip()]
            if vals:
                named[k] = list(dict.fromkeys([*named.get(k, []), *vals]))[:20]
    save_profile(company_id, p, reason)


def answer_question(company_id: int, qid: int, answer: str) -> dict:
    qrow = db.one("SELECT * FROM interview_questions WHERE id=? AND company_id=?", (qid, company_id))
    if not qrow:
        raise ValueError("question not found")
    answer = answer.strip()
    db.update("interview_questions", qid, {"answer": answer, "status": "answered", "answered_at": db.now(),
                                           "updated_at": db.now()})
    p = get_profile(company_id)
    data, used = llm.chat_json(FOLLOW_SYSTEM, FOLLOW_PROMPT.format(
        profile=profile_summary_for_prompt(company_id, p, 2500), history=_history(company_id),
        area=FIELD_LABEL.get(qrow["area"], qrow["area"]), question=qrow["question"], answer=answer),
        purpose="interview_followup", temperature=0.3)
    statements = _rule_extract(answer)
    entities = None
    followup = ""
    reason = ""
    if isinstance(data, dict):
        statements = [clean_ws(str(s)) for s in data.get("extracted", []) if str(s).strip()] or statements
        entities = data.get("entities") if isinstance(data.get("entities"), dict) else None
        followup = clean_ws(str(data.get("followup") or ""))
        reason = clean_ws(str(data.get("followup_reason") or ""))
    else:
        # Rule-based follow-up: vague answer, or a newly named organisation worth monitoring.
        caps = [m for m in re.findall(r"\b([A-Z][a-zA-Z0-9&]+(?:\s[A-Z][a-zA-Z0-9&]+)*)\b", answer)
                if m.lower() not in p.get("name", "").lower() and len(m) > 2 and m not in ("I", "We", "Our", "The")]
        if len(answer.split()) < 8:
            followup = f"Could you be a little more specific about that? For example, numbers, names or recent examples for: {truncate(qrow['question'], 90)}"
            reason = "answer was very short"
        elif qrow["area"] in ("competitors", "market_changes", "material_risks") and caps:
            followup = f"You mentioned {caps[0]}. What exactly about {caps[0]} should we watch most closely?"
            reason = "answer named an entity worth monitoring"
    _apply_to_profile(company_id, qrow, statements, entities, f"interview answer: {qrow['area']}")

    created_followup = None
    n_follow = db.scalar("SELECT COUNT(*) FROM interview_questions WHERE company_id=? AND is_followup=1", (company_id,))
    if followup and not qrow["is_followup"] and n_follow < MAX_FOLLOWUPS and not _is_repeat(company_id, followup):
        fid = db.insert("interview_questions", {"company_id": company_id, "seq": qrow["seq"], "parent_id": qid,
                                                "area": qrow["area"], "question": followup, "rationale": reason,
                                                "status": "pending", "is_followup": 1, "generated_by": used,
                                                "updated_at": db.now()})
        created_followup = fid
    else:
        _refine_next(company_id)
    return {"followup_id": created_followup, "extracted": statements, "by": used}


def _is_repeat(company_id: int, text: str) -> bool:
    for r in ordered(company_id):
        if fuzz.token_set_ratio(r["question"], text) >= 88:
            return True
    return False


def _refine_next(company_id: int) -> None:
    nxt = next((r for r in ordered(company_id) if r["status"] == "pending" and not r["asked_at"]), None)
    if not nxt or nxt["is_followup"]:
        return
    answered = [r for r in ordered(company_id) if r["status"] == "answered"]
    if not answered:
        return
    p = get_profile(company_id)
    data, used = llm.chat_json(REFINE_SYSTEM, REFINE_PROMPT.format(
        name=p.get("name"), history=_history(company_id, 8), area=FIELD_LABEL.get(nxt["area"]), question=nxt["question"]),
        purpose="interview_refine", temperature=0.3, max_tokens=400)
    if isinstance(data, dict) and data.get("question"):
        newq = clean_ws(str(data["question"]))
        others = [r for r in ordered(company_id) if r["id"] != nxt["id"]]
        if 15 < len(newq) < 400 and all(fuzz.token_set_ratio(o["question"], newq) < 88 for o in others):
            db.update("interview_questions", nxt["id"], {"question": newq, "rationale": (nxt["rationale"] or "") +
                                                         " | adapted to earlier answers", "generated_by": used,
                                                         "updated_at": db.now()})


def skip_question(company_id: int, qid: int) -> None:
    qrow = db.one("SELECT * FROM interview_questions WHERE id=? AND company_id=?", (qid, company_id))
    if not qrow:
        raise ValueError("question not found")
    db.update("interview_questions", qid, {"status": "skipped", "answer": None, "answered_at": db.now(),
                                           "updated_at": db.now()})
    p = get_profile(company_id)
    p.setdefault("skipped_areas", [])
    if qrow["area"] not in p["skipped_areas"]:
        p["skipped_areas"].append(qrow["area"])
    save_profile(company_id, p, f"question skipped (confidential): {qrow['area']}")
    _refine_next(company_id)


def edit_answer(company_id: int, qid: int, answer: str) -> None:
    qrow = db.one("SELECT * FROM interview_questions WHERE id=? AND company_id=?", (qid, company_id))
    if not qrow:
        raise ValueError("question not found")
    answer = answer.strip()
    db.update("interview_questions", qid, {"answer": answer, "status": "answered" if answer else "skipped",
                                           "updated_at": db.now()})
    statements = _rule_extract(answer) if answer else []
    data, _ = llm.chat_json(FOLLOW_SYSTEM, FOLLOW_PROMPT.format(
        profile=profile_summary_for_prompt(company_id, None, 2000), history="(edited answer)",
        area=FIELD_LABEL.get(qrow["area"]), question=qrow["question"], answer=answer),
        purpose="interview_edit", temperature=0.2) if answer else (None, "")
    entities = None
    if isinstance(data, dict):
        statements = [clean_ws(str(s)) for s in data.get("extracted", []) if str(s).strip()] or statements
        entities = data.get("entities") if isinstance(data.get("entities"), dict) else None
    _apply_to_profile(company_id, qrow, statements, entities, f"answer edited: {qrow['area']}")


def finish_if_complete(company_id: int) -> bool:
    pr = progress(company_id)
    if pr["complete"]:
        c = db.one("SELECT status FROM companies WHERE id=?", (company_id,))
        if c and c["status"] in ("researched", "interviewing"):
            db.update("companies", company_id, {"status": "interviewed", "updated_at": db.now()})
    return pr["complete"]
