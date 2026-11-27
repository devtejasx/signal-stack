"""Polite HTTP fetcher.

* Identifies itself honestly (config.USER_AGENT) - never impersonates a browser.
* Obeys robots.txt (including Crawl-delay) for every host it touches.
* Rate-limits per host and backs off on 429/503.
* Never attempts to get past logins, CAPTCHAs or bot challenges: such responses are
  recorded as "blocked" and the source is flagged, not retried with tricks.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from . import config

log = logging.getLogger("signalstack.fetch")

# Hosts with published API rate rules stricter than our default.
HOST_DELAYS = {
    "api.gdeltproject.org": 5.5,
    "hn.algolia.com": 1.0,
    "api.stackexchange.com": 1.0,
    "en.wikipedia.org": 1.0,
    "www.bing.com": 3.0,
    "itunes.apple.com": 3.0,
}

CHALLENGE_MARKERS = ("captcha", "cf-challenge", "challenge-platform", "are you a robot", "unusual traffic",
                     "/cdn-cgi/challenge", "anomaly-modal")


@dataclass
class FetchResult:
    url: str
    status: int = 0
    text: str = ""
    headers: dict = field(default_factory=dict)
    final_url: str = ""
    error: str = ""
    blocked: str = ""          # robots | challenge | auth
    latency_ms: int = 0
    not_modified: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and not self.blocked and not self.error


class Fetcher:
    def __init__(self) -> None:
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()
        self._host_next: dict[str, float] = {}
        self._host_lock = threading.Lock()
        self.client = httpx.Client(
            headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en;q=0.9"},
            timeout=config.HTTP_TIMEOUT,
            follow_redirects=True,
        )

    # -- robots ---------------------------------------------------------------
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self._robots_lock:
            if base in self._robots:
                return self._robots[base]
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(base + "/robots.txt")
        parsed: urllib.robotparser.RobotFileParser | None = rp
        try:
            self._wait_host(p.netloc)
            r = self.client.get(base + "/robots.txt", timeout=12)
            ctype = r.headers.get("content-type", "")
            if r.status_code in (401, 403):
                rp.disallow_all = True
            elif r.status_code >= 400 or "html" in ctype.lower():
                rp.allow_all = True  # no robots file (or an HTML error page served instead) => everything allowed
            else:
                rp.parse(r.text.splitlines())
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError, ValueError):
            rp.allow_all = True  # unreachable/invalid host: the page fetch itself will report the failure
        with self._robots_lock:
            self._robots[base] = parsed
        return parsed

    def allowed(self, url: str) -> bool:
        rp = self._robots_for(url)
        if rp is None:
            return True
        try:
            return rp.can_fetch(config.USER_AGENT, url) and rp.can_fetch("SignalStackBot", url)
        except Exception:  # noqa: BLE001
            return True

    def _crawl_delay(self, url: str) -> float:
        host = urlparse(url).netloc
        d = HOST_DELAYS.get(host, config.DEFAULT_HOST_DELAY)
        rp = self._robots.get(f"{urlparse(url).scheme}://{host}")
        if rp is not None:
            try:
                cd = rp.crawl_delay("SignalStackBot") or rp.crawl_delay("*")
                if cd:
                    d = max(d, min(float(cd), 30.0))
            except Exception:  # noqa: BLE001
                pass
        return d

    def _wait_host(self, host: str, delay: float | None = None) -> None:
        delay = HOST_DELAYS.get(host, config.DEFAULT_HOST_DELAY) if delay is None else delay
        with self._host_lock:
            nxt = self._host_next.get(host, 0.0)
            now = time.monotonic()
            wait = max(0.0, nxt - now)
            self._host_next[host] = max(now, nxt) + delay
        if wait:
            time.sleep(wait)

    # -- fetch ----------------------------------------------------------------
    def get(self, url: str, *, etag: str | None = None, last_modified: str | None = None,
            check_robots: bool = True, accept: str | None = None, params: dict | None = None) -> FetchResult:
        res = FetchResult(url=url)
        if check_robots and not self.allowed(url):
            res.blocked = "robots"
            res.error = "Disallowed by robots.txt"
            return res
        host = urlparse(url).netloc
        self._wait_host(host, self._crawl_delay(url))
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        if accept:
            headers["Accept"] = accept
        t0 = time.monotonic()
        for attempt in range(3):
            try:
                r = self.client.get(url, headers=headers, params=params)
            except (httpx.InvalidURL, UnicodeError, ValueError) as e:  # malformed / over-long host names
                res.error = f"Invalid URL: {e}"[:300]
                return res
            except httpx.HTTPError as e:
                res.error = f"{type(e).__name__}: {e}"[:300]
                dns_fail = isinstance(e, httpx.ConnectError) and ("getaddrinfo" in str(e) or "Name or service" in str(e))
                if attempt < 2 and not dns_fail:
                    time.sleep(2 + attempt * 3)
                    continue
                res.latency_ms = int((time.monotonic() - t0) * 1000)
                return res
            if r.status_code in (429, 503) and attempt < 2:
                retry = r.headers.get("retry-after", "")
                time.sleep(min(int(retry) if retry.isdigit() else 6 * (attempt + 1), 30))
                continue
            break
        res.latency_ms = int((time.monotonic() - t0) * 1000)
        res.status = r.status_code
        res.final_url = str(r.url)
        res.headers = {k.lower(): v for k, v in r.headers.items()}
        res.error = ""
        if r.status_code == 304:
            res.not_modified = True
            return res
        if r.status_code in (401, 407):
            res.blocked = "auth"
            res.error = "Requires authentication - not bypassed"
            return res
        content = r.content[: config.MAX_BYTES]
        try:
            res.text = content.decode(r.encoding or "utf-8", errors="replace")
        except LookupError:
            res.text = content.decode("utf-8", errors="replace")
        low = res.text[:20000].lower()
        if r.status_code in (403, 202, 429) and any(m in low for m in CHALLENGE_MARKERS):
            res.blocked = "challenge"
            res.error = "Bot challenge returned - source not crawled (no bypass attempted)"
        elif r.status_code >= 400:
            res.error = f"HTTP {r.status_code}"
        return res


_fetcher: Fetcher | None = None
_fetcher_lock = threading.Lock()


def fetcher() -> Fetcher:
    global _fetcher
    with _fetcher_lock:
        if _fetcher is None:
            _fetcher = Fetcher()
        return _fetcher
