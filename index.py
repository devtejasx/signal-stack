"""Vercel entrypoint (serverless). Locally and on Render use `uvicorn signalstack.web:app` instead.

Serverless functions have a read-only filesystem except /tmp and no long-running process, so:
* the demo database (deploy/seed.db) is copied to /tmp on a cold start,
* the in-process scheduler is off; Vercel Cron calls /api/cron/* on the weekly / monthly schedule (vercel.json).
Data written on the hosted demo lives until the instance is recycled.
"""
import os
import shutil
from pathlib import Path

os.environ.setdefault("SIGNALSTACK_DB", "/tmp/signalstack/signalstack.db")
os.environ.setdefault("SIGNALSTACK_UPLOADS", "/tmp/signalstack/uploads")
os.environ.setdefault("SIGNALSTACK_OUTBOX", "/tmp/signalstack/outbox")
os.environ.setdefault("ENABLE_SCHEDULER", "0")
os.environ.setdefault("CRON_TOKEN", os.environ.get("CRON_SECRET", ""))  # Vercel Cron sends "Bearer $CRON_SECRET"

_db = Path(os.environ["SIGNALSTACK_DB"])
_seed = Path(__file__).parent / "deploy" / "seed.db"
if not _db.exists() and _seed.exists():
    _db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_seed, _db)

from signalstack.web import app  # noqa: E402,F401
