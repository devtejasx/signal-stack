"""Thin SQLite data-access layer (stdlib sqlite3, WAL mode, thread-local connections)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


def init_db() -> None:
    global _initialised
    with _init_lock:
        if _initialised:
            return
        schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        conn().executescript(schema)
        conn().commit()
        _initialised = True


def q(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    cur = conn().execute(sql, tuple(params))
    return [dict(r) for r in cur.fetchall()]


def one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    cur = conn().execute(sql, tuple(params))
    r = cur.fetchone()
    return dict(r) if r else None


def scalar(sql: str, params: Iterable[Any] = ()) -> Any:
    cur = conn().execute(sql, tuple(params))
    r = cur.fetchone()
    return r[0] if r else None


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    c = conn()
    cur = c.execute(sql, tuple(params))
    c.commit()
    return cur.lastrowid


def insert(table: str, row: dict) -> int:
    cols = ", ".join(row.keys())
    ph = ", ".join("?" for _ in row)
    vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in row.values()]
    return execute(f"INSERT INTO {table} ({cols}) VALUES ({ph})", vals)


def update(table: str, row_id: int, fields: dict, key: str = "id") -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in fields.values()]
    execute(f"UPDATE {table} SET {sets} WHERE {key}=?", [*vals, row_id])


def loads(v: str | None, default: Any = None) -> Any:
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default
