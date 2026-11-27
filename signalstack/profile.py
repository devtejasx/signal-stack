"""Company intelligence profile: storage, versioning and the 'interest document'
that every relevance score is computed against."""
from __future__ import annotations

import json

from . import db

INTEL_FIELDS: list[tuple[str, str]] = [
    ("business_lines", "Business lines & revenue streams"),
    ("priority_segments", "Highest-value customer segments"),
    ("customer_problems", "Customer problems & unmet needs"),
    ("differentiators", "Why customers choose us over alternatives"),
    ("competitors", "Competitors & substitutes (founder view)"),
    ("market_changes", "Industry & market changes"),
    ("material_risks", "Developments that could materially affect us"),
    ("growth_priorities", "Growth priorities, next 6-12 months"),
    ("internal_data", "Internal information that should influence decisions"),
    ("decisions", "Decisions the intelligence system should support"),
]
FIELD_LABEL = dict(INTEL_FIELDS)


def get_profile(company_id: int) -> dict:
    row = db.one("SELECT data FROM profiles WHERE company_id=?", (company_id,))
    p = db.loads(row["data"], {}) if row else {}
    p.setdefault("intelligence", {})
    return p


def save_profile(company_id: int, profile: dict, reason: str) -> int:
    row = db.one("SELECT version FROM profiles WHERE company_id=?", (company_id,))
    version = (row["version"] + 1) if row else 1
    data = json.dumps(profile, ensure_ascii=False)
    if row:
        db.execute("UPDATE profiles SET data=?, version=?, updated_at=? WHERE company_id=?",
                   (data, version, db.now(), company_id))
    else:
        db.execute("INSERT INTO profiles(company_id, data, version, updated_at) VALUES (?,?,?,?)",
                   (company_id, data, version, db.now()))
    db.insert("profile_history", {"company_id": company_id, "version": version, "data": data, "reason": reason,
                                  "created_at": db.now()})
    return version


def answered_context(company_id: int) -> list[dict]:
    return db.q("SELECT id, area, question, answer, status, is_followup FROM interview_questions "
                "WHERE company_id=? AND status IN ('answered','skipped') ORDER BY seq, id", (company_id,))


def interest_document(company_id: int, profile: dict | None = None) -> str:
    """Bag of text describing what this company cares about. Used for TF-IDF relevance."""
    p = profile or get_profile(company_id)
    parts: list[str] = [p.get("name", ""), p.get("summary", ""), p.get("industry", "")]
    for k in ("offerings", "customer_segments", "geographies", "value_props", "category_keywords",
              "customer_problem_queries", "market_topics", "regulatory_topics"):
        parts += [str(x) for x in p.get(k, []) or []]
    for k, vals in (p.get("intelligence") or {}).items():
        for v in vals or []:
            parts.append(v.get("text", "") if isinstance(v, dict) else str(v))
    comps = db.q("SELECT name FROM competitors WHERE company_id=? AND status='approved'", (company_id,))
    parts += [c["name"] for c in comps]
    # Weight the founder's own words more heavily (interview answers twice).
    for a in answered_context(company_id):
        if a["status"] == "answered" and a["answer"]:
            parts += [a["answer"], a["answer"]]
    return "\n".join(x for x in parts if x)


def profile_summary_for_prompt(company_id: int, profile: dict | None = None, max_chars: int = 5000) -> str:
    p = profile or get_profile(company_id)
    lines = [f"Company: {p.get('name')} ({p.get('website')})", f"Summary: {p.get('summary')}",
             f"Industry: {p.get('industry')}", f"Business model: {p.get('business_model')}",
             f"Offerings: {', '.join(p.get('offerings', [])[:10])}",
             f"Customer segments: {', '.join(p.get('customer_segments', [])[:8])}",
             f"Geographies: {', '.join(p.get('geographies', [])[:10])}"]
    intel = p.get("intelligence") or {}
    for k, label in INTEL_FIELDS:
        vals = [v.get("text") if isinstance(v, dict) else str(v) for v in intel.get(k, [])]
        if vals:
            lines.append(f"{label} (founder): {' | '.join(vals)[:600]}")
    comps = db.q("SELECT name FROM competitors WHERE company_id=? AND status='approved'", (company_id,))
    if comps:
        lines.append("Tracked competitors: " + ", ".join(c["name"] for c in comps))
    s = "\n".join(lines)
    return s[:max_chars]
