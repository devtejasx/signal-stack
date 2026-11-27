"""Monitoring layer: visit approved sources, collect new records, de-duplicate, keep history."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz

from . import adapters, db
from .jobs import JobContext
from .textutil import age_days, canonical_url, clean_ws, domain_of, sha

log = logging.getLogger("signalstack.monitor")
MAX_RECORD_AGE_DAYS = 365  # older records are history, not intelligence


def _check_source(source: dict) -> tuple[dict, adapters.AdapterResult | None, str]:
    try:
        return source, adapters.run_adapter(source), ""
    except Exception as e:  # noqa: BLE001
        return source, None, f"{type(e).__name__}: {e}"[:300]


def run_crawl(company_id: int, trigger: str, ctx: JobContext | None = None, source_ids: list[int] | None = None) -> int:
    step = ctx.step if ctx else (lambda m: None)
    run_id = db.insert("crawl_runs", {"company_id": company_id, "trigger": trigger, "started_at": db.now(), "status": "running"})
    sql = "SELECT * FROM sources WHERE company_id=? AND status='approved' AND type!='upload'"
    sources = db.q(sql, (company_id,))
    if source_ids:
        sources = [s for s in sources if s["id"] in source_ids]
    step(f"Crawl run #{run_id}: checking {len(sources)} approved sources")
    stats = {"checked": 0, "found": 0, "new": 0, "dup": 0}
    errors = []
    # Different hosts are fetched in parallel; the fetcher enforces per-host politeness.
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_check_source, s) for s in sources]
        for fut in as_completed(futures):
            source, res, exc = fut.result()
            stats["checked"] += 1
            if res is None:
                res = adapters.AdapterResult()
                res.status, res.error = "error", exc
            new = dup = 0
            if res.status == "ok":
                for rec in res.records:
                    outcome = _store_record(company_id, source, run_id, rec)
                    new += outcome == "new"
                    dup += outcome == "dup"
            stats["found"] += len(res.records)
            stats["new"] += new
            stats["dup"] += dup
            ok = res.status in ("ok", "not_modified")
            db.insert("source_checks", {"run_id": run_id, "source_id": source["id"], "checked_at": db.now(),
                                        "status": res.status, "http_status": res.http_status, "latency_ms": res.latency_ms,
                                        "items_found": len(res.records), "items_new": new, "error": res.error or None})
            fields = {"last_checked": db.now(), "last_status": res.status,
                      "consecutive_failures": 0 if ok else source["consecutive_failures"] + 1,
                      "items_total": source["items_total"] + new}
            if ok:
                fields["last_success"] = db.now()
            if new:
                fields["last_new_item_at"] = db.now()
            if res.etag:
                fields["etag"] = res.etag
            if res.last_modified:
                fields["last_modified"] = res.last_modified
            db.update("sources", source["id"], fields)
            if not ok:
                errors.append({"source_id": source["id"], "name": source["name"], "status": res.status, "error": res.error})
            step(f"  {source['name'][:60]}: {res.status}, {len(res.records)} records, {new} new")
    db.update("crawl_runs", run_id, {"finished_at": db.now(), "status": "success" if len(errors) < len(sources) or not sources else "failed",
                                     "sources_checked": stats["checked"], "items_found": stats["found"],
                                     "items_new": stats["new"], "items_duplicate": stats["dup"], "errors": errors})
    step(f"Crawl run #{run_id} done: {stats['new']} new records, {stats['dup']} duplicates, {len(errors)} source errors")
    return run_id


def _store_record(company_id: int, source: dict, run_id: int, rec: dict) -> str:
    title = clean_ws(rec.get("title"))
    url = rec.get("url") or ""
    if not title or not url:
        return "skip"
    a = age_days(rec.get("published_at"))
    if a is not None and a > MAX_RECORD_AGE_DAYS:
        return "skip"
    uid = rec.get("uid") or url
    url_hash = sha(canonical_url(uid) if uid.startswith("http") else uid)
    if db.scalar("SELECT 1 FROM items WHERE company_id=? AND url_hash=?", (company_id, url_hash)):
        return "seen"
    content_hash = sha((title + " " + (rec.get("snippet") or "")).lower())
    # Near-duplicate: same story from another publisher/query => corroboration, not a new development.
    dup_of = None
    recent = db.q("SELECT id, title, publisher, content_hash FROM items WHERE company_id=? AND dup_of IS NULL "
                  "AND discovered_at >= datetime('now','-45 days') ORDER BY id DESC LIMIT 1500", (company_id,))
    for r in recent:
        if r["content_hash"] == content_hash or (len(title) > 25 and fuzz.token_set_ratio(r["title"], title) >= 92):
            dup_of = r["id"]
            if (r["publisher"] or "") != (rec.get("publisher") or ""):
                db.execute("UPDATE items SET corroborations = corroborations + 1 WHERE id=?", (r["id"],))
            break
    extra = rec.get("extra") or {}
    db.execute(
        "INSERT OR IGNORE INTO items (company_id, source_id, run_id, category, competitor_id, url, url_hash, title, snippet, "
        "publisher, published_at, discovered_at, content_hash, dup_of, analysis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (company_id, source["id"], run_id, source["category"], source.get("competitor_id"), url, url_hash, title[:500],
         rec.get("snippet") or "", rec.get("publisher") or domain_of(url), rec.get("published_at"), db.now(), content_hash,
         dup_of, __import__("json").dumps({"extra": extra}) if extra else None))
    return "dup" if dup_of else "new"
