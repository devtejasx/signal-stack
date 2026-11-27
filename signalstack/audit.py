"""Monthly Signal Stack audit: source health, duplicates, inactivity, declining relevance,
newly discovered sources, add/remove recommendations, monthly source-health report."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import db
from .fetch import fetcher
from .jobs import JobContext
from .textutil import age_days


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def run_audit(company_id: int, trigger: str = "manual", ctx: JobContext | None = None, discover_new: bool = True) -> int:
    step = ctx.step if ctx else (lambda m: None)
    company = db.one("SELECT * FROM companies WHERE id=?", (company_id,))
    sources = db.q("SELECT * FROM sources WHERE company_id=? AND status IN ('approved','discovered','blocked') AND type!='upload'",
                   (company_id,))
    now = datetime.now(timezone.utc)
    d30 = (now - timedelta(days=30)).isoformat()
    d90 = (now - timedelta(days=90)).isoformat()
    report = {"company": company["name"], "generated_at": db.now(), "sources_checked": len(sources), "dead": [], "failing": [],
              "inactive": [], "duplicates": [], "declining": [], "healthy": [], "blocked": [], "new_sources": [],
              "recommend_remove": [], "recommend_add": [], "recommend_review": []}

    step(f"Audit: checking {len(sources)} sources for dead URLs")
    f = fetcher()
    for s in sources:
        entry = {"id": s["id"], "name": s["name"], "category": s["category"], "status": s["status"], "url": s["url"]}
        if s["status"] == "blocked":
            report["blocked"].append({**entry, "why": (db.loads(s["probe"], {}) or {}).get("error") or s["last_status"]})
            continue
        if s["type"] in ("rss", "webpage", "site_updates") and s["url"]:
            r = f.get(s["url"])
            if r.blocked == "robots":
                report["blocked"].append({**entry, "why": "robots.txt now disallows this URL"})
                continue
            if not r.ok and (r.status >= 400 or r.status == 0):
                report["dead"].append({**entry, "why": r.error or f"HTTP {r.status}"})
                continue
        if s["consecutive_failures"] >= 3:
            report["failing"].append({**entry, "why": f"{s['consecutive_failures']} consecutive failed checks (last: {s['last_status']})"})
        if s["status"] == "approved":
            created_age = age_days(s["created_at"]) or 0
            last_new = age_days(s["last_new_item_at"]) if s["last_new_item_at"] else None
            if created_age > 21 and (last_new is None or last_new > 30):
                report["inactive"].append({**entry, "why": "no new records in the last 30 days" if last_new else "has never produced a record"})
            recent = db.one("SELECT AVG(relevance) r, COUNT(*) n FROM items WHERE source_id=? AND discovered_at>=?", (s["id"], d30))
            older = db.one("SELECT AVG(relevance) r, COUNT(*) n FROM items WHERE source_id=? AND discovered_at<? AND discovered_at>=?",
                           (s["id"], d30, d90))
            if recent["n"] >= 3 and older["n"] >= 3 and recent["r"] is not None and older["r"] is not None and recent["r"] < older["r"] - 0.12:
                report["declining"].append({**entry, "why": f"avg relevance fell from {older['r']:.2f} to {recent['r']:.2f}"})
            sig = db.scalar("SELECT COUNT(*) FROM items WHERE source_id=? AND significant=1", (s["id"],))
            tot = db.scalar("SELECT COUNT(*) FROM items WHERE source_id=?", (s["id"],))
            if tot >= 15 and sig == 0:
                report["declining"].append({**entry, "why": f"{tot} records collected, none ever significant (low signal-to-noise)"})

    step("Audit: detecting duplicate / overlapping sources")
    approved = [s for s in sources if s["status"] == "approved"]
    url_sets = {s["id"]: {r["url_hash"] for r in db.q("SELECT url_hash FROM items WHERE source_id=?", (s["id"],))} for s in approved}
    seen_pairs = set()
    for a in approved:
        for b in approved:
            if a["id"] >= b["id"] or (a["id"], b["id"]) in seen_pairs:
                continue
            same_def = (a["type"] == b["type"] and (a["url"] or "") == (b["url"] or "") and (a["query"] or "").lower() == (b["query"] or "").lower())
            # overlap via cross-source near-duplicates as well as identical URLs
            dup_items = db.scalar("SELECT COUNT(*) FROM items x JOIN items y ON x.dup_of=y.id WHERE x.source_id=? AND y.source_id=?",
                                  (b["id"], a["id"])) or 0
            jac = _jaccard(url_sets[a["id"]], url_sets[b["id"]])
            smaller = min(len(url_sets[a["id"]]), len(url_sets[b["id"]])) or 1
            overlap = max(jac, dup_items / smaller)
            if same_def or overlap >= 0.5:
                seen_pairs.add((a["id"], b["id"]))
                keep, drop = (a, b) if (a["overall_score"] or 0) >= (b["overall_score"] or 0) else (b, a)
                report["duplicates"].append({"keep": {"id": keep["id"], "name": keep["name"]},
                                             "drop": {"id": drop["id"], "name": drop["name"]},
                                             "overlap": round(1.0 if same_def else overlap, 2)})
                db.update("sources", drop["id"], {"duplicate_of": keep["id"]})

    flagged = {e["id"] for k in ("dead", "failing", "inactive", "declining") for e in report[k]} | \
              {d["drop"]["id"] for d in report["duplicates"]}
    report["healthy"] = [{"id": s["id"], "name": s["name"], "category": s["category"]} for s in approved if s["id"] not in flagged]
    for k, label in (("dead", "dead URL"), ("failing", "repeated failures"), ("inactive", "inactive")):
        for e in report[k]:
            report["recommend_remove"].append({**e, "reason": f"{label}: {e['why']}"})
    for d in report["duplicates"]:
        report["recommend_remove"].append({"id": d["drop"]["id"], "name": d["drop"]["name"],
                                           "reason": f"duplicate of '{d['keep']['name']}' (overlap {d['overlap']})"})
    for e in report["declining"]:
        report["recommend_review"].append({**e, "reason": f"declining relevance: {e['why']}"})

    if discover_new:
        step("Audit: running source discovery for new candidates")
        before = {s["id"] for s in db.q("SELECT id FROM sources WHERE company_id=?", (company_id,))}
        from .discovery import build_signal_stack
        try:
            build_signal_stack(company_id, ctx, origin="audit")
        except Exception as e:  # noqa: BLE001 - discovery failure must not kill the health report
            report["discovery_error"] = str(e)[:300]
        new = db.q("SELECT * FROM sources WHERE company_id=?", (company_id,))
        for s in new:
            if s["id"] in before:
                continue
            if s["status"] == "approved":  # audit never auto-approves: founder decides
                db.update("sources", s["id"], {"status": "discovered", "approved_at": None})
            entry = {"id": s["id"], "name": s["name"], "category": s["category"], "score": s["overall_score"], "reason": s["reason"]}
            report["new_sources"].append(entry)
            if (s["overall_score"] or 0) >= 0.55:
                report["recommend_add"].append(entry)

    md = _markdown(report)
    aid = db.insert("audits", {"company_id": company_id, "created_at": db.now(), "trigger": trigger, "report": report, "markdown": md})
    step(f"Audit #{aid}: {len(report['healthy'])} healthy, {len(report['recommend_remove'])} to remove, "
         f"{len(report['recommend_add'])} to add, {len(report['duplicates'])} duplicate pairs")
    return aid


def _markdown(r: dict) -> str:
    L = [f"# Monthly Source-Health Report — {r['company']}", f"*Generated {r['generated_at'][:16]} UTC · {r['sources_checked']} sources checked*", ""]
    L.append(f"**Healthy:** {len(r['healthy'])} · **Dead:** {len(r['dead'])} · **Failing:** {len(r['failing'])} · "
             f"**Inactive:** {len(r['inactive'])} · **Duplicates:** {len(r['duplicates'])} · **Declining relevance:** {len(r['declining'])} · "
             f"**Access-restricted:** {len(r['blocked'])} · **Newly discovered:** {len(r['new_sources'])}")
    for key, title in (("recommend_remove", "Recommended removals"), ("recommend_review", "Review (declining relevance)"),
                       ("recommend_add", "Recommended additions")):
        L += ["", f"## {title}"]
        if not r[key]:
            L.append("None.")
        for e in r[key]:
            L.append(f"- **{e['name']}** ({e.get('category', '')}) — {e.get('reason', '')}")
    L += ["", "## Duplicate / overlapping sources"]
    L += [f"- keep **{d['keep']['name']}**, drop **{d['drop']['name']}** (overlap {d['overlap']})" for d in r["duplicates"]] or ["None."]
    L += ["", "## Access-restricted sources (not crawled)"]
    L += [f"- **{e['name']}** — {e['why']}" for e in r["blocked"]] or ["None."]
    return "\n".join(L)
