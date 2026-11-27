"""Two independent automated processes per company (plus optional interim monitoring):

A. weekly_brief   - cron: settings.briefing_day @ briefing_time in settings.timezone
B. monthly_audit  - cron: settings.audit_day_of_month @ audit_time
C. monitor        - every settings.monitor_every_hours (collects + analyses between briefs)

APScheduler runs inside the web process (or headless via `python -m signalstack.cli scheduler`).
Every run, success or failure, is written to job_runs; missed runs are logged as 'missed'.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import config, db, pipeline
from .jobs import run_job

log = logging.getLogger("signalstack.scheduler")
_sched: BackgroundScheduler | None = None


def _run(company_id: int, job: str):
    makers = {"weekly_brief": pipeline.weekly_brief_job, "monthly_audit": pipeline.audit_job, "monitor": pipeline.monitor_job}
    run_job(company_id, job, "schedule", makers[job](company_id))


def _on_missed(event):
    try:
        job, cid = event.job_id.split(":")[0], int(event.job_id.split(":")[1])
        db.insert("job_runs", {"company_id": cid, "job": job, "trigger": "schedule",
                               "scheduled_for": event.scheduled_run_time.isoformat(), "started_at": db.now(),
                               "finished_at": db.now(), "status": "missed",
                               "error": "Scheduled run was missed (process was not running at the scheduled time)."})
    except Exception:  # noqa: BLE001
        log.exception("could not log missed job")


def start() -> BackgroundScheduler:
    global _sched
    if _sched is None:
        _sched = BackgroundScheduler(job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600})
        _sched.add_listener(_on_missed, EVENT_JOB_MISSED)
        _sched.start()
    sync_jobs()
    try:
        check_missed_runs(catch_up=config.CATCH_UP_MISSED)
    except Exception:  # noqa: BLE001 - never block start-up
        log.exception("missed-run check failed")
    return _sched


def _last_fire(trigger, lookback_days: int) -> datetime | None:
    now = datetime.now(timezone.utc)
    t = trigger.get_next_fire_time(None, now - timedelta(days=lookback_days))
    last = None
    while t and t <= now:
        last = t
        t = trigger.get_next_fire_time(t, t + timedelta(seconds=1))
    return last


def check_missed_runs(catch_up: bool = True) -> list[dict]:
    """A scheduled run that fell while the process was down (laptop asleep, free host idling) leaves no trace in
    APScheduler. On start-up compare each company's last scheduled fire time with job_runs, log a 'missed' row so it
    shows in the dashboard, and (optionally) run it now."""
    missed = []
    for s in db.q("SELECT s.*, c.created_at AS ccreated FROM settings s JOIN companies c ON c.id=s.company_id "
                  "WHERE c.status IN ('stack_ready','monitoring')"):
        cid, tz = s["company_id"], s["timezone"] or "UTC"
        checks = []
        if s["brief_enabled"]:
            h, m = (s["briefing_time"] or "07:30").split(":")
            checks.append(("weekly_brief", CronTrigger(day_of_week=s["briefing_day"], hour=int(h), minute=int(m), timezone=tz), 8))
        if s["audit_enabled"]:
            h, m = (s["audit_time"] or "06:00").split(":")
            checks.append(("monthly_audit", CronTrigger(day=int(s["audit_day_of_month"] or 1), hour=int(h), minute=int(m), timezone=tz), 32))
        for job, trig, days in checks:
            last = _last_fire(trig, days)
            if not last or last.isoformat() < (s["ccreated"] or ""):
                continue
            since = (last - timedelta(minutes=5)).isoformat()
            # any run of this kind (scheduled, manual or demo) at/after the fire time satisfies it
            kinds = ("weekly_brief", "manual_brief") if job == "weekly_brief" else ("monthly_audit",)
            ran = db.one(f"SELECT 1 FROM job_runs WHERE company_id=? AND job IN ({','.join('?' * len(kinds))}) "
                         "AND started_at >= ? AND status IN ('success','running','missed')", (cid, *kinds, since))
            if ran:
                continue
            db.insert("job_runs", {"company_id": cid, "job": job, "trigger": "schedule", "scheduled_for": last.isoformat(),
                                   "started_at": db.now(), "finished_at": db.now(), "status": "missed",
                                   "error": "The app was not running at the scheduled time"
                                            + ("; caught up automatically on start-up." if catch_up else ".")})
            missed.append({"company_id": cid, "job": job, "scheduled_for": last.isoformat()})
            if catch_up:
                from .jobs import run_in_background
                makers = {"weekly_brief": pipeline.weekly_brief_job, "monthly_audit": pipeline.audit_job}
                run_in_background(cid, job, "catch_up", makers[job](cid, "catch_up"))
    return missed


def sync_jobs() -> None:
    if _sched is None:
        return
    for job in _sched.get_jobs():
        job.remove()
    for s in db.q("SELECT s.*, c.status AS cstatus FROM settings s JOIN companies c ON c.id=s.company_id"):
        cid = s["company_id"]
        if s["cstatus"] not in ("stack_ready", "monitoring", "interviewed", "interviewing"):
            continue
        tz = s["timezone"] or "UTC"
        if s["brief_enabled"]:
            h, m = (s["briefing_time"] or "07:30").split(":")
            _sched.add_job(_run, CronTrigger(day_of_week=s["briefing_day"], hour=int(h), minute=int(m), timezone=tz),
                           args=[cid, "weekly_brief"], id=f"weekly_brief:{cid}", replace_existing=True)
        if s["audit_enabled"]:
            h, m = (s["audit_time"] or "06:00").split(":")
            _sched.add_job(_run, CronTrigger(day=int(s["audit_day_of_month"] or 1), hour=int(h), minute=int(m), timezone=tz),
                           args=[cid, "monthly_audit"], id=f"monthly_audit:{cid}", replace_existing=True)
        if s["monitor_enabled"] and s["monitor_every_hours"]:
            _sched.add_job(_run, IntervalTrigger(hours=int(s["monitor_every_hours"]), timezone=tz),
                           args=[cid, "monitor"], id=f"monitor:{cid}", replace_existing=True)


def next_runs(company_id: int) -> dict:
    out = {}
    if _sched is None:
        return out
    for j in _sched.get_jobs():
        name, cid = j.id.split(":")
        if int(cid) == company_id and j.next_run_time:
            out[name] = j.next_run_time.isoformat()
    return out


def running() -> bool:
    return _sched is not None and _sched.running


def shutdown() -> None:
    global _sched
    if _sched is not None:
        _sched.shutdown(wait=False)
        _sched = None
