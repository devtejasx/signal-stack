"""Runtime configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p  # relative paths are relative to the project root, not the cwd


DB_PATH = _path("SIGNALSTACK_DB", "data/signalstack.db")
UPLOAD_DIR = _path("SIGNALSTACK_UPLOADS", "data/uploads")
OUTBOX_DIR = _path("SIGNALSTACK_OUTBOX", "data/outbox")

# Honest, identifiable crawler user agent. We never impersonate a browser.
USER_AGENT = os.getenv(
    "SIGNALSTACK_USER_AGENT",
    "SignalStackBot/1.0 (company-intelligence research prototype; respects robots.txt)",
)
DEFAULT_HOST_DELAY = float(os.getenv("SIGNALSTACK_HOST_DELAY", "2.0"))  # seconds between hits on one host
HTTP_TIMEOUT = float(os.getenv("SIGNALSTACK_HTTP_TIMEOUT", "20"))
MAX_BYTES = int(os.getenv("SIGNALSTACK_MAX_BYTES", str(3_000_000)))

# --- LLM providers (all free / free-tier / local) -------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | groq | gemini | openrouter | ollama | pollinations | none
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Free-tier limits are per model, so several models form a fallback chain (best first). Models the account
# cannot use are filtered out at runtime via GET /models.
GROQ_MODELS = os.getenv("GROQ_MODELS", os.getenv("GROQ_MODEL", "") or
                        "openai/gpt-oss-120b,qwen/qwen3.8-27b,llama-3.3-70b-versatile,moonshotai/kimi-k2-instruct-0905,"
                        "moonshotai/kimi-k2-instruct,meta-llama/llama-4-maverick-17b-128e-instruct,"
                        "meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3-32b,openai/gpt-oss-20b,llama-3.1-8b-instant")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OLLAMA_URL = os.getenv("OLLAMA_URL", "")  # e.g. http://localhost:11434
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "10240"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "3600"))  # CPU inference can be slow
POLLINATIONS_ENABLED = _bool("POLLINATIONS_ENABLED", True)  # keyless free endpoint
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai")
# Anonymous tier: 1 reply / 45 s per IP and ~800 output tokens -> last-resort fallback only.
POLLINATIONS_MIN_INTERVAL = float(os.getenv("POLLINATIONS_MIN_INTERVAL", "46"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# Optional official-API credentials for sources that forbid anonymous crawling.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

# Optional SMTP delivery for the Monday brief (otherwise .eml files go to the outbox).
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")

INTERPRET_LIMIT = int(os.getenv("INTERPRET_LIMIT", "30"))  # LLM-interpreted signals per run
AUTO_APPROVE_THRESHOLD = float(os.getenv("AUTO_APPROVE_THRESHOLD", "0.55"))
ENABLE_SCHEDULER = _bool("ENABLE_SCHEDULER", True)
# On start-up, log scheduled runs that fell while the app was down and run them now.
CATCH_UP_MISSED = _bool("CATCH_UP_MISSED", True)
# Shared secret for POST /api/cron/{job}: lets a free external scheduler (e.g. GitHub Actions) wake an idle host
# and trigger the weekly brief / monthly audit. Empty = endpoint disabled.
CRON_TOKEN = os.getenv("CRON_TOKEN", "")

for _d in (DB_PATH.parent, UPLOAD_DIR, OUTBOX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
