"""Capture dashboard screenshots as test evidence (headless Chrome/Edge, no extra packages).

    python scripts/capture_evidence.py [base_url] [chrome_path]

Expects the app to be running (default http://localhost:8766). Writes evidence/<company>_<page>.png
plus evidence/INDEX.md listing what each screenshot proves.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evidence"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766"
CANDIDATES = [sys.argv[2]] if len(sys.argv) > 2 else [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser"]

PAGES = [  # (path suffix, file label, height, what it proves)
    ("", "overview", 1500, "Company overview: pipeline status, counts, top signals, next scheduled runs"),
    ("/profile", "profile", 2600, "Intelligence profile: verified facts (with source links) vs assumptions; editable founder answers"),
    ("/interview", "interview", 3000, "Adaptive 10-question interview with follow-ups, skips and edit controls"),
    ("/stack", "signal_stack", 3200, "Signal Stack: verified competitors + Customers / Competitors / Market / Internal sources with reasons"),
    ("/sources", "source_manager", 3000, "Source Manager: approve / reject / remove / add, health, duplicates, uploads"),
    ("/feed", "feed", 3000, "Intelligence feed with filters (category, competitor, date, relevance, impact, verification)"),
    ("/briefs", "brief_history", 1100, "Brief history (historical records) and manual trigger"),
    ("/audits", "audit", 2600, "Monthly source-health audit report"),
    ("/runs", "runs", 2200, "Runs & Failures: every scheduled/manual run, crawl history"),
    ("/settings", "settings", 1400, "Settings: briefing day/time, time zone, monitoring and refresh frequency, monthly audit"),
]


def chrome() -> str:
    for c in CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    sys.exit("Chrome/Edge not found - pass its path as the 2nd argument")


def shot(exe: str, url: str, path: Path, height: int) -> bool:
    with tempfile.TemporaryDirectory() as prof:
        r = subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                            f"--window-size=1440,{height}", f"--screenshot={path}", "--virtual-time-budget=4000", url],
                           capture_output=True, timeout=120)
    return path.exists() and path.stat().st_size > 5000


def main() -> None:
    OUT.mkdir(exist_ok=True)
    exe = chrome()
    home = httpx.get(BASE + "/", timeout=60).text
    companies = list(dict.fromkeys(re.findall(r'href="/c/(\d+)"><b>([^<]+)</b>', home)))
    lines = ["# Test evidence — dashboard screenshots", "",
             f"Captured from a running instance ({BASE}) backed by the demo database. Each image is a real page of the app.", "",
             "| Screenshot | What it shows |", "|---|---|"]
    ok = shot(exe, BASE + "/", OUT / "00_home.png", 900)
    lines.append(f"| [00_home.png](00_home.png) | Company setup + configured companies{'' if ok else ' (capture failed)'} |")
    for cid, name in companies:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        pages = PAGES + []
        bid = re.findall(r'/c/%s/briefs/(\d+)"' % cid, httpx.get(f"{BASE}/c/{cid}/briefs", timeout=60).text)
        if bid:
            pages.append((f"/briefs/{bid[0]}", "monday_brief", 9000, "Latest Monday Morning Brief (10 sections, evidence links, action tracking)"))
        for suffix, label, h, what in pages:
            f = OUT / f"{slug}_{label}.png"
            ok = shot(exe, f"{BASE}/c/{cid}{suffix}", f, h)
            lines.append(f"| [{f.name}]({f.name}) | **{name}** — {what}{'' if ok else ' (capture failed)'} |")
            print(("ok  " if ok else "FAIL"), f.name)
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
