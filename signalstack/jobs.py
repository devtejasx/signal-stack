"""Job runner: every scheduled or manual pipeline run is logged in ``job_runs``.

Failures are captured with their traceback and surfaced on the dashboard
("Runs & Failures" page) - nothing fails silently.
"""
from __future__ import annotations

import logging
import threading
import traceback
from typing import Callable

from . import db

log = logging.getLogger("signalstack.jobs")
_running: dict[tuple[int | None, str], int] = {}
_lock = threading.Lock()


class JobContext:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self._lines: list[str] = []

    def step(self, msg: str) -> None:
        log.info("[job %s] %s", self.job_id, msg)
        self._lines.append(f"{db.now()}  {msg}")
        db.update("job_runs", self.job_id, {"progress": msg, "log": "\n".join(self._lines[-400:])})


def run_job(company_id: int | None, job: str, trigger: str, fn: Callable[[JobContext], object],
            scheduled_for: str | None = None) -> dict:
    """Run synchronously, logging start/finish/failure. Returns the job_runs row."""
    key = (company_id, job)
    with _lock:
        if key in _running:
            existing = _running[key]
            return {"id": existing, "status": "already_running"}
        job_id = db.insert("job_runs", {"company_id": company_id, "job": job, "trigger": trigger,
                                        "scheduled_for": scheduled_for, "started_at": db.now(), "status": "running"})
        _running[key] = job_id
    ctx = JobContext(job_id)
    try:
        result = fn(ctx)
        db.update("job_runs", job_id, {"status": "success", "finished_at": db.now(),
                                       "result_ref": str(result) if result is not None else None})
    except Exception as e:  # noqa: BLE001 - we want every failure recorded
        tb = traceback.format_exc()
        log.exception("job %s failed", job)
        db.update("job_runs", job_id, {"status": "failed", "finished_at": db.now(),
                                       "error": f"{type(e).__name__}: {e}\n\n{tb[-3000:]}"})
    finally:
        with _lock:
            _running.pop(key, None)
    return db.one("SELECT * FROM job_runs WHERE id=?", (job_id,))


def run_in_background(company_id: int | None, job: str, trigger: str, fn: Callable[[JobContext], object]) -> int | None:
    """Start a job on a daemon thread and return its job id (or the running one)."""
    with _lock:
        if (company_id, job) in _running:
            return _running[(company_id, job)]
    holder: dict = {}
    started = threading.Event()

    def _target():
        def wrapped(ctx: JobContext):
            holder["id"] = ctx.job_id
            started.set()
            return fn(ctx)
        run_job(company_id, job, trigger, wrapped)
        started.set()

    threading.Thread(target=_target, daemon=True, name=f"job-{job}-{company_id}").start()
    started.wait(timeout=5)
    return holder.get("id")


def is_running(company_id: int | None, job: str | None = None) -> bool:
    with _lock:
        return any(k[0] == company_id and (job is None or k[1] == job) for k in _running)


def mark_stale_runs() -> None:
    """On startup, runs left 'running' by a crashed process are marked failed (visible on dashboard)."""
    db.execute("UPDATE job_runs SET status='failed', finished_at=?, error=COALESCE(error,'') || "
               "'Process stopped before the job finished (server restart or crash).' WHERE status='running'",
               (db.now(),))
