"""Command-line entry points.

    python -m signalstack.cli add "MeetUniversity" https://www.meetuniversity.com
    python -m signalstack.cli research <company_id>
    python -m signalstack.cli plan <company_id>          # print the interview plan
    python -m signalstack.cli stack <company_id>
    python -m signalstack.cli crawl <company_id>
    python -m signalstack.cli brief <company_id>
    python -m signalstack.cli audit <company_id>
    python -m signalstack.cli scheduler                  # run the scheduler headless
"""
from __future__ import annotations

import logging
import sys
import time

from . import db, interview, pipeline, scheduler
from .jobs import run_job


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_db()
    if not argv:
        print(__doc__)
        return 1
    cmd, args = argv[0], argv[1:]
    if cmd == "add":
        print(pipeline.create_company(args[0], args[1]))
    elif cmd == "research":
        print(run_job(int(args[0]), "research", "cli", pipeline.research_job(int(args[0]))))
    elif cmd == "plan":
        for q in interview.plan_interview(int(args[0])):
            print(f"[{q['area']}] {q['question']}\n    ({q['rationale']}) — {q['generated_by']}")
    elif cmd == "stack":
        print(run_job(int(args[0]), "discovery", "cli", pipeline.discovery_job(int(args[0]))))
    elif cmd == "crawl":
        print(run_job(int(args[0]), "monitor", "cli", pipeline.monitor_job(int(args[0]), "manual")))
    elif cmd == "brief":
        print(run_job(int(args[0]), "manual_brief", "cli", pipeline.weekly_brief_job(int(args[0]), "manual")))
    elif cmd == "audit":
        print(run_job(int(args[0]), "monthly_audit", "cli", pipeline.audit_job(int(args[0]), "manual")))
    elif cmd == "scheduler":
        scheduler.start()
        print("Scheduler running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                scheduler.sync_jobs()
        except KeyboardInterrupt:
            scheduler.shutdown()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
