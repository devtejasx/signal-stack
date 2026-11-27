"""Reproduce the demonstration end to end for the companies in demo/companies.json.

    python scripts/run_demo.py                 # all companies
    python scripts/run_demo.py meetuniversity  # one company

Steps per company: create -> research -> adaptive interview (answers keyed by area, see the
note in companies.json) -> Signal Stack -> crawl -> (synthetic internal upload, if configured)
-> analysis -> Monday brief -> monthly audit -> (business-context change, if configured).
Everything runs through the same code paths as the dashboard; every step is logged in job_runs.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signalstack import analysis, context, db, interview, pipeline  # noqa: E402
from signalstack.internal import ingest_upload  # noqa: E402
from signalstack.jobs import run_job  # noqa: E402
from signalstack.profile import get_profile, save_profile  # noqa: E402
from signalstack.textutil import domain_of  # noqa: E402

log = logging.getLogger("demo")


def run_interview(cid: int, answers: dict) -> None:
    interview.plan_interview(cid)
    for _ in range(30):
        q = interview.next_question(cid)
        if not q:
            break
        key = f"{q['area']}_followup" if q["is_followup"] else q["area"]
        ans = answers.get(key) or answers.get(q["area"], "")
        if ans == "SKIP" or not ans:
            log.info("SKIP  [%s] %s", q["area"], q["question"])
            interview.skip_question(cid, q["id"])
        else:
            log.info("Q [%s]%s %s", q["area"], " (follow-up)" if q["is_followup"] else "", q["question"])
            interview.answer_question(cid, q["id"], ans)
    interview.finish_if_complete(cid)


def second_pass(cid: int, fixes: list[dict]) -> None:
    """Week 2: the founder reviews the Signal Stack and supplies official websites for competitors the automatic lookup
    could not confirm (same code path as Signal Stack -> add competitor). Every one is re-verified live, then a second
    crawl and a second brief run - which also demonstrates the historical crawl/brief records."""
    p = get_profile(cid)
    for f in fixes:
        p.setdefault("competitor_candidates", []).append({"name": f["name"], "domain": domain_of(f["website"]),
                                                          "why": f["why"], "kind": "direct"})
        p.setdefault("founder_named", {}).setdefault("competitors", []).append(f["name"])
        save_profile(cid, p, f"competitor added: {f['name']}")
    run_job(cid, "discovery", "demo", pipeline.discovery_job(cid))
    run_job(cid, "monitor", "demo", pipeline.monitor_job(cid, "demo"))
    run_job(cid, "manual_brief", "demo", pipeline.weekly_brief_job(cid, "demo"))


def run_company(spec: dict) -> int:
    cid = pipeline.create_company(spec["name"], spec["website"])
    done = lambda job: db.one("SELECT 1 FROM job_runs WHERE company_id=? AND job=? AND status='success'", (cid, job))  # noqa: E731
    if not done("research"):
        run_job(cid, "research", "demo", pipeline.research_job(cid))
    run_interview(cid, spec["answers"])
    if not done("discovery"):
        run_job(cid, "discovery", "demo", pipeline.discovery_job(cid))
    if not done("monitor"):
        run_job(cid, "monitor", "demo", pipeline.monitor_job(cid, "demo"))
    if spec.get("synthetic_upload") and not db.one("SELECT 1 FROM uploads WHERE company_id=?", (cid,)):
        path = ROOT / spec["synthetic_upload"]
        ingest_upload(cid, path.name, path.read_bytes(), "crm_export", is_synthetic=True)
        analysis.score_items(cid)
    if not done("manual_brief"):
        run_job(cid, "manual_brief", "demo", pipeline.weekly_brief_job(cid, "demo"))
    if not done("monthly_audit"):
        run_job(cid, "monthly_audit", "demo", pipeline.audit_job(cid, "demo"))
    if spec.get("founder_competitor_fixes") and not db.one("SELECT 1 FROM profile_history WHERE company_id=? AND reason LIKE 'competitor added:%'", (cid,)):
        second_pass(cid, spec["founder_competitor_fixes"])
    if spec.get("context_change") and not db.one("SELECT 1 FROM stack_revisions WHERE company_id=?", (cid,)):
        run_job(cid, "context_change", "demo", lambda ctx: context.apply_context_change(cid, spec["context_change"], ctx))
    return cid


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    db.init_db()
    specs = json.loads((ROOT / "demo" / "companies.json").read_text(encoding="utf-8"))["companies"]
    for spec in specs:
        if argv and spec["key"] not in argv:
            continue
        log.info("=== %s ===", spec["name"])
        run_company(spec)


if __name__ == "__main__":
    main(sys.argv[1:])
