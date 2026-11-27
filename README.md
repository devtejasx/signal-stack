# Signal Stack — Company Intelligence & Executive Briefing Engine

Enter a company name and website. Signal Stack researches the business (separating verified facts from assumptions),
runs an adaptive 10-question founder interview, builds a justified **Signal Stack** across **Customers, Competitors,
Market and Internal** intelligence, crawls those sources on a schedule, ranks what matters and writes a
**Monday Morning Brief** a founder can act on — with every development traceable to a source link.

Built for the EDXSO AI Engineer Intern assignment. Everything is free / open source: the demo uses open-weights models on the
**Groq free tier** (no card), with local Ollama and a rule-based path as fallbacks. No paid APIs, no paid scraping, no bypassing
of robots.txt or bot protection.

| | |
|---|---|
| Sample reports | [`reports/`](reports/) — MeetUniversity, Zerodha, Licious (brief, interview, profile, Signal Stack, audit) |
| Minimum-target checklist | [`reports/MINIMUM_TARGETS.md`](reports/MINIMUM_TARGETS.md) (computed from the live database) |
| Technical documentation | [`docs/TECHNICAL.md`](docs/TECHNICAL.md) |
| Cost statement | [`docs/COST_STATEMENT.md`](docs/COST_STATEMENT.md) |
| Test evidence | [`evidence/`](evidence/) (screenshots) and `tests/` |
| Demo video script | [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) |

## What it does

1. **Company setup** — name + URL.
2. **Initial research** — homepage and key pages (robots.txt respected), sitemap structure, Wikipedia, recent news,
   App Store presence. Facts are only called facts if they match the text of a fetched page; everything else is an
   assumption to confirm.
3. **Adaptive AI interview** — 10 questions generated from that company's research (one per intelligence area), asked one
   at a time, with follow-ups, re-worded next questions, skip for confidential topics, editable answers, and a structured
   intelligence profile that updates after every answer.
4. **Signal Stack** — competitors are *discovered and verified* (live homepage similarity), never hard-coded. Each candidate
   source is probed once and scored for **specificity, recency and credibility**; the reason for monitoring is written from that
   evidence. Sources above the threshold are approved; the rest wait for the founder in the Source Manager.
5. **Monitoring** — RSS/Atom, news search (Bing News RSS, GDELT), App Store reviews, Stack Exchange, Hacker News, page-change
   detection on competitor pricing/positioning pages, new posts on competitor blogs, government feeds, and uploads (CSV, Excel,
   PDF, text) for internal data. Duplicates are folded into the original as corroboration; full crawl history is kept.
6. **Analysis & ranking** — separate measures for *strategic relevance*, *source reliability* and *evidence strength*, plus
   impact, urgency, recency and novelty, with a plain-English explanation per signal. The LLM interprets the top signals (what
   happened / why it matters for *this* company / opportunity or threat / urgency / what to consider) and that interpretation is
   stored and shown separately from the evidence. The AI's confidence never counts as proof.
7. **Monday Morning Brief** — the 10 required sections; every interpretive statement must cite signal IDs that exist (uncited
   statements are dropped); empty categories say so explicitly. Export: Markdown, printable HTML/PDF, email-ready HTML, JSON;
   delivery by SMTP or `.eml` outbox; action tracking in the dashboard.
8. **Automation** — weekly brief (cron per company: day, time, time zone), monthly source-health audit (dead, failing,
   inactive, duplicate, declining-relevance sources; newly discovered sources; add/remove recommendations), interim monitoring,
   manual triggers, and a Runs & Failures page — failed or missed scheduled runs are shown on every page.
9. **Context change** — describe how the business changed; the system updates the profile, re-scores every source against the
   new context, retires sources that no longer fit, discovers new ones and writes a before/after explanation.

## Run it locally (Windows / macOS / Linux)

Requirements: Python 3.11+ and a free LLM key — [Groq](https://console.groq.com/keys) (recommended, no card) or
[Gemini](https://aistudio.google.com/apikey) — or a local [Ollama](https://ollama.com). Without any LLM the app still runs in a
rule-based mode.

```bash
git clone <this repo> && cd edxso-signal-stack
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # then set GROQ_API_KEY=... (free) — or run a local model:
# ollama pull qwen2.5:7b && set OLLAMA_URL=http://localhost:11434 and OLLAMA_MODEL=qwen2.5:7b in .env

uvicorn signalstack.web:app --port 8000
# open http://localhost:8000  → enter "MeetUniversity" + https://www.meetuniversity.com
```

### Deploy (free)

`render.yaml` is a Render Blueprint: on render.com choose **New → Blueprint**, pick this repository, paste `GROQ_API_KEY` when
asked and deploy (free plan). The first boot seeds the demo database (`deploy/seed.db`). Free-tier limits: the instance sleeps
after ~15 minutes idle and the disk is ephemeral. The in-process scheduler catches up missed runs on wake-up, and
`.github/workflows/scheduled-runs.yml` (repo secrets `APP_URL`, `CRON_TOKEN`) wakes the app on Monday / the 1st of the month and
calls `POST /api/cron/weekly_brief|monthly_audit`.

Docker (app + Ollama): `docker compose up --build`, then `docker compose exec ollama ollama pull qwen2.5:3b`.

### Reproduce the demonstration

```bash
python scripts/run_demo.py            # MeetUniversity, Zerodha, Licious: research → interview → stack → crawl → brief → audit (+ context change for Licious)
python scripts/export_reports.py      # writes reports/<company>/*.md|html and reports/MINIMUM_TARGETS.md
python -m pytest -q                   # unit tests
```

CLI for single steps: `python -m signalstack.cli add|research|plan|stack|crawl|brief|audit|scheduler`.

## Repository layout

```
signalstack/
  web.py            FastAPI dashboard + JSON API          templates/, static/   server-rendered UI
  research.py       company research, facts vs assumptions interview.py          adaptive interview
  discovery.py      competitor verification, source discovery & probing
  adapters.py       source adapters (rss, news, gdelt, reviews, Q&A, page diff, blog updates, reddit*)
  fetch.py          polite fetcher: robots.txt, per-host rate limits, honest user agent, no bypassing
  monitor.py        crawl runs, de-duplication, corroboration, health tracking
  analysis.py       ranking (separate reliability / evidence / relevance) + LLM interpretation
  brief.py          Monday Morning Brief (evidence layer + cited interpretation) and renderers
  internal.py       CSV / Excel / PDF / text ingestion, labelling, clustering
  audit.py          monthly source-health audit          context.py   business-context change re-assessment
  scheduler.py      APScheduler jobs                     jobs.py      job log, failures, background runs
  llm.py            free/local LLM providers + JSON repair + cache   schema.sql   database schema
scripts/            run_demo.py, export_reports.py
demo/               demo founder answers (from public info) + a SYNTHETIC CRM file for the internal pipeline
reports/            generated sample reports              docs/        technical doc, cost statement, demo script
```

## Honesty notes

* Demo founder answers were written by the candidate from each company's public website and public information — the
  companies' founders were not interviewed. They are labelled in the interview transcripts.
* The MeetUniversity internal-intelligence section is driven by a **synthetic** CRM file
  (`demo/SYNTHETIC_crm_export_meetuniversity.csv`) to demonstrate the pipeline; it is flagged as synthetic in the brief.
* All external developments in the briefs come from real, linked sources collected by the crawler.
