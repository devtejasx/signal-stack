"""Provider-agnostic LLM access restricted to free / free-tier / local models.

Order of preference in ``auto`` mode:
    Groq (free tier key) -> Gemini (free tier key) -> OpenRouter (":free" models)
    -> Ollama (local, open-source) -> Pollinations (keyless free endpoint)
If none is reachable every caller falls back to a deterministic rule-based path,
so the product keeps working with zero AI services.

The LLM is only ever used for *interpretation* (questions, profile synthesis,
"why it matters", recommendations). Facts and evidence always come from fetched
sources and are verified outside the model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass

import httpx

from . import config, db

log = logging.getLogger("signalstack.llm")


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    min_interval: float = 0.0
    extra_headers: dict | None = None


_lock = threading.Lock()
_last_call: dict[str, float] = {}
_disabled_until: dict[str, float] = {}
_groq_available: list[str] | None = None

# Prompts that contain the company's *internal* data (CRM exports, call notes) go to the local model
# when one is configured, so confidential data never leaves the machine.
LOCAL_FIRST_PURPOSES = {"internal_themes"}


def _groq_models() -> list[str]:
    """Configured Groq models, filtered to the ones the account can actually use (free-tier limits are per model,
    so rotating across several models spreads the daily token budget)."""
    global _groq_available
    wanted = [m.strip() for m in config.GROQ_MODELS.split(",") if m.strip()]
    if _groq_available is None:
        try:
            r = httpx.get("https://api.groq.com/openai/v1/models", timeout=15,
                          headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"})
            r.raise_for_status()
            _groq_available = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        except Exception as e:  # noqa: BLE001
            log.warning("Could not list Groq models (%s); using the configured list as-is", e)
            return wanted
    return [m for m in wanted if m in _groq_available] or _groq_available[:3]


def _key(p: Provider) -> str:
    return f"{p.name}:{p.model}"


def providers(purpose: str = "") -> list[Provider]:
    ps: list[Provider] = []
    want = config.LLM_PROVIDER.lower()
    if want == "none":
        return []
    if config.GROQ_API_KEY and want in ("auto", "groq"):
        for m in _groq_models():
            ps.append(Provider("groq", "https://api.groq.com/openai/v1", m, config.GROQ_API_KEY, 2.1))
    if config.GEMINI_API_KEY and want in ("auto", "gemini"):
        ps.append(Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", config.GEMINI_MODEL,
                           config.GEMINI_API_KEY, 6.5))
    if config.OPENROUTER_API_KEY and want in ("auto", "openrouter"):
        ps.append(Provider("openrouter", "https://openrouter.ai/api/v1", config.OPENROUTER_MODEL,
                           config.OPENROUTER_API_KEY, 3.5))
    if config.OLLAMA_URL and want in ("auto", "ollama"):
        ps.append(Provider("ollama", config.OLLAMA_URL.rstrip("/") + "/v1", config.OLLAMA_MODEL))
    if config.POLLINATIONS_ENABLED and want in ("auto", "pollinations"):
        ps.append(Provider("pollinations", "https://text.pollinations.ai", config.POLLINATIONS_MODEL, "",
                           config.POLLINATIONS_MIN_INTERVAL))
    if purpose in LOCAL_FIRST_PURPOSES:
        ps.sort(key=lambda p: p.name != "ollama")
    return ps


def active_provider_label() -> str:
    ps = [p for p in providers() if _disabled_until.get(_key(p), 0) <= time.monotonic()] or providers()
    return _key(ps[0]) if ps else "rule-based (no LLM configured)"


def _throttle(p: Provider) -> None:
    with _lock:
        last = _last_call.get(_key(p), 0.0)
        wait = p.min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call[_key(p)] = time.monotonic()


class ProviderError(RuntimeError):
    """kind: rate_limited (retry same model after `wait`), skip_model (try the next model), fatal."""

    def __init__(self, kind: str, msg: str, wait: float = 0.0):
        super().__init__(f"{kind}: {msg}")
        self.kind, self.wait = kind, wait


def _parse_wait(text: str, header: str | None) -> float:
    try:
        if header:
            return float(header)
    except ValueError:
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", text)
    if m and any(m.groups()):
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    return 10.0


def _is_reasoning(model: str) -> bool:
    return model.startswith(("openai/gpt-oss", "qwen/qwen3")) or "deepseek-r1" in model


def _endpoint(p: Provider) -> str:
    if p.name == "pollinations":
        return "https://text.pollinations.ai/openai"
    return p.base_url + "/chat/completions"


def _call_ollama(p: Provider, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool) -> str:
    """Native Ollama API: lets us raise the context window (default 2-4k is too small) and force JSON."""
    body = {"model": p.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": config.OLLAMA_NUM_CTX}}
    if json_mode:
        body["format"] = "json"
    r = httpx.post(config.OLLAMA_URL.rstrip("/") + "/api/chat", json=body, timeout=config.OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def _call(p: Provider, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool = False) -> str:
    if p.name == "ollama":
        return _call_ollama(p, messages, temperature, max_tokens, json_mode)
    headers = {"Content-Type": "application/json", "User-Agent": config.USER_AGENT}
    if p.api_key:
        headers["Authorization"] = f"Bearer {p.api_key}"
    body = {"model": p.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if p.name == "pollinations":
        body["private"] = True  # do not list the prompt in the public feed
    if p.name == "groq":
        if _is_reasoning(p.model):
            # reasoning tokens count against max_tokens; keep them small and out of the answer
            body["max_tokens"] = max_tokens + 1000
            if p.model.startswith("openai/gpt-oss"):
                body["reasoning_effort"] = config.GROQ_REASONING_EFFORT
                body["include_reasoning"] = False
            else:
                body["reasoning_format"] = "hidden"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
    _throttle(p)
    r = httpx.post(_endpoint(p), headers=headers, json=body, timeout=config.LLM_TIMEOUT)
    txt = r.text[:600]
    if r.status_code == 429 or (r.status_code == 402 and "FREE_LIMIT" in r.text):
        wait = _parse_wait(txt, r.headers.get("retry-after"))
        if re.search(r"per day|\(TPD\)|\(RPD\)", txt) or wait > 90:
            raise ProviderError("skip_model", f"daily quota reached ({txt[:160]})", wait)
        raise ProviderError("rate_limited", txt[:160], wait)
    if r.status_code in (401, 403) and p.name != "pollinations":
        raise ProviderError("skip_model", f"key rejected by {p.name} (HTTP {r.status_code}) - check the API key", 86400)
    if r.status_code == 503:
        raise ProviderError("rate_limited", "service unavailable", 12.0)
    if r.status_code == 413 or (r.status_code == 400 and re.search(r"too large|context_length|maximum context", txt, re.I)):
        raise ProviderError("skip_model", f"request too large for {p.model}")
    if r.status_code in (400, 404) and re.search(r"model_not_found|decommissioned|does not exist|not supported", txt, re.I):
        raise ProviderError("skip_model", f"model unavailable: {txt[:160]}", 86400)
    if r.status_code == 400 and json_mode and "json_validate_failed" in txt:
        return _call(p, messages, temperature, max_tokens, json_mode=False)  # let extract_json repair the text
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"] or ""


def chat(system: str, user: str, *, purpose: str = "", temperature: float = 0.2, max_tokens: int = 2500,
         use_cache: bool = True, json_mode: bool = False) -> tuple[str | None, str]:
    """Return (text, provider_label). text is None when no provider succeeded."""
    ps = providers(purpose)
    if not ps:
        return None, "rule-based"
    key = hashlib.sha256(json.dumps([system, user, temperature], sort_keys=True).encode()).hexdigest()
    if use_cache:
        cached = db.one("SELECT response, provider FROM llm_cache WHERE prompt_hash=?", (key,))
        if cached:
            return cached["response"], cached["provider"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for p in ps:
        if _disabled_until.get(_key(p), 0) > time.monotonic():
            continue
        for attempt in range(4):
            t0 = time.monotonic()
            try:
                text = _call(p, msgs, temperature, max_tokens, json_mode)
                ms = int((time.monotonic() - t0) * 1000)
                db.insert("llm_calls", {"purpose": purpose, "provider": p.name, "model": p.model, "prompt_hash": key,
                                        "ok": 1, "latency_ms": ms, "created_at": db.now()})
                label = _key(p)
                if text.strip():
                    db.execute("INSERT OR REPLACE INTO llm_cache(prompt_hash, provider, response, created_at) VALUES (?,?,?,?)",
                               (key, label, text, db.now()))
                    return text, label
                break  # empty answer: try the next model
            except Exception as e:  # noqa: BLE001 - provider errors are logged and we fall through
                ms = int((time.monotonic() - t0) * 1000)
                db.insert("llm_calls", {"purpose": purpose, "provider": p.name, "model": p.model, "prompt_hash": key,
                                        "ok": 0, "latency_ms": ms, "error": str(e)[:300], "created_at": db.now()})
                log.warning("LLM %s failed (%s): %s", _key(p), purpose, e)
                if isinstance(e, ProviderError):
                    if e.kind == "rate_limited":
                        time.sleep(min(60.0, e.wait + 1) * (1 + 0.5 * attempt))
                        continue
                    if e.kind == "skip_model":
                        if e.wait:
                            _disabled_until[_key(p)] = time.monotonic() + e.wait
                        if "key rejected" in str(e):  # the key is bad for every model of this provider
                            for q in ps:
                                if q.name == p.name:
                                    _disabled_until[_key(q)] = time.monotonic() + e.wait
                        break
                if isinstance(e, httpx.ReadTimeout) and p.name == "ollama":
                    break  # a local model that timed out once will time out again; fall through
                if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    _disabled_until[_key(p)] = time.monotonic() + 300
                    break
                time.sleep(2 * (attempt + 1))
    return None, "rule-based"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str | None):
    if not text:
        return None
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            if start < 0:
                continue
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cand[start:i + 1])
                        except ValueError:
                            break
    return None


def repair_truncated_json(text: str):
    """Recover the longest valid prefix of a JSON object cut off by an output-token limit."""
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    stack: list[str] = []
    in_str = esc = False
    cuts: list[tuple[int, str]] = []
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    for pos, closers in reversed(cuts[-400:]):
        try:
            data = json.loads(s[:pos] + closers)
            if isinstance(data, dict):
                return data
        except ValueError:
            continue
    return None


def chat_json(system: str, user: str, *, purpose: str = "", temperature: float = 0.2, max_tokens: int = 3000):
    """Ask for JSON; returns (parsed_or_None, provider_label)."""
    sys = system + "\n\nRespond with valid JSON only. No markdown, no commentary."
    text, label = chat(sys, user, purpose=purpose, temperature=temperature, max_tokens=max_tokens, json_mode=True)
    data = extract_json(text)
    if text is not None and not isinstance(data, dict) and text.lstrip().lstrip("`json \n").startswith("{"):
        data = repair_truncated_json(text)  # output was cut off by a token limit
    if data is None and text is not None:
        # one repair attempt, uncached
        text2, label = chat(sys, user + "\n\nYour previous answer was not valid JSON. Return ONLY the JSON.",
                            purpose=purpose + ":retry", temperature=0.0, max_tokens=max_tokens, use_cache=False,
                            json_mode=True)
        data = extract_json(text2)
    return data, (label if data is not None else "rule-based")
