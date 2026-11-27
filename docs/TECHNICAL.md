# Signal Stack — Technical Documentation

*Company Intelligence & Executive Briefing Engine · EDXSO AI Engineer Intern, Assignment 1*

## 1. Architecture

```
            ┌──────────────── FastAPI dashboard (Jinja2, no build step) ────────────────┐
 founder ──►│ Setup · Profile · Interview · Signal Stack · Sources · Feed · Brief · Audit │
            └───────┬───────────────────────────────────────────────────────────┬───────┘
                    │ background jobs (job_runs log)                            │ APScheduler
      ┌─────────────▼───────────┐  ┌─────────────────┐  ┌──────────────────┐   │ weekly brief (cron)
      │ research.py             │  │ interview.py    │  │ discovery.py     │   │ monthly audit (cron)
      │ site + sitemap + wiki + │─►│ 10 planned Qs,  │─►│ verified         │   │ monitoring (interval)
      │ news + app store        │  │ follow-ups,     │  │ competitors +    │   │
      │ facts vs assumptions    │  │ profile updates │  │ probed sources   │   │
      └─────────────────────────┘  └─────────────────┘  └────────┬─────────┘   │
                                                                  ▼             │
  adapters.py (rss · bing_news · gdelt · hn · stackexchange · appstore_reviews · webpage-diff · site_updates · reddit*)
                                                                  │  fetch.py: robots.txt, per-host rate limits,
                                                                  ▼  honest UA, no bypass of challenges/logins
  monitor.py ──► items (dedupe: URL hash + fuzzy title → corroboration) ──► analysis.py (scores + interpretation)
                                                                  │
                                         brief.py (evidence layer + cited interpretation) ──► md / html / email / json
                                         audit.py (source health)   context.py (re-assessment on context change)
  Storage: SQLite (schema.sql, WAL).   LLM: llm.py — free/local providers only, rule-based fallback.
```

* **Language/stack:** Python 3.12, FastAPI, Jinja2, SQLite (stdlib `sqlite3`), APScheduler, httpx, BeautifulSoup/lxml,
  feedparser, scikit-learn (TF-IDF, cosine similarity, KMeans), VADER sentiment, rapidfuzz, pandas/openpyxl/pypdf.
* **Why this shape:** a single process with an embedded scheduler is the simplest thing that satisfies "working scheduler +
  functioning backend" and runs anywhere (laptop, Docker, a free VM). SQLite keeps every crawl, check, record, brief and audit
  as history with zero ops. The UI is server-rendered so there is no separate front-end build to break.
* **AI is optional, evidence is not:** every fact and every reported development comes from a fetched source and is stored with
  its URL, publication date, discovery date and snippet. The LLM only produces *interpretation*, which is stored and displayed
  separately. If no LLM is reachable, every step has a deterministic fallback and the product still runs.

## 2. Company understanding & adaptive interview

1. **Research** (`research.py`): homepage + up to 8 key pages chosen by URL/link-text patterns (about, services, pricing,
   customers, news, FAQ, careers, contact), the sitemap (section sizes reveal what the business really publishes), Wikipedia
   (confirmed when the article links the official domain), recent news mentions, and App Store presence (iTunes Search API).
2. **Facts vs assumptions:** verbatim page elements (title, meta description, headings, sitemap counts, app ratings) are facts.
   LLM-extracted "facts" are re-verified by token overlap against the page they cite (≥60 % match) — otherwise they are
   downgraded to assumptions. Inferences (revenue model, segments) are always assumptions until the founder confirms them.
3. **Interview** (`interview.py`): 10 questions, one per intelligence area (business lines, segments, problems,
   differentiators, competitors, market changes, material risks, growth priorities, internal data, decisions), worded around
   what research found or could not confirm. After every answer the system (a) extracts structured statements into the
   profile, (b) decides on a follow-up (max one per question, four per interview), (c) re-words the next question to build on
   previous answers. Fuzzy matching (rapidfuzz ≥ 88) prevents repeated questions. Skips are recorded as confidential; every
   answer is editable later and edits re-score all signals.

## 3. Source discovery (no hard-coded lists)

* **Competitors:** candidates from (i) research synthesis, (ii) founder-named competitors, (iii) organisations that actually
  appear in category news headlines. Founder-named names without a website get their official domain proposed by the model and
  then checked live; names that are generic substitutes ("local agents", "neighbourhood butchers") become market-news topics
  instead of fake competitor profiles. Each candidate is **verified on evidence fetched from its live homepage**: TF-IDF cosine
  similarity with the company's own pages (shared terms stored as evidence) **and** a model verdict — *direct / substitute /
  adjacent / unrelated* — that may only use the fetched homepage text. Direct and substitute competitors are approved (max 10),
  adjacent ones wait for the founder, unrelated, parked or unreachable domains are rejected with the reason stored. A
  founder-named competitor whose site exists but refuses bots (HTTP 403 / challenge) is kept and monitored via news only.
* **Candidate sources per category**, generated from the profile:
  * *Customers* — the company's own and competitors' App Store reviews (real customer language), news on customer-problem
    queries, Stack Exchange communities **selected by measuring hits** for the company's problem queries across a catalogue of
    sites, Hacker News (kept only if the probe shows recent relevant threads), subreddits (listed, but blocked unless official
    API credentials are configured — Reddit's robots.txt forbids crawling).
  * *Competitors* — per verified competitor: news query, change-detection on its pricing/positioning page (primary source),
    and new posts on its blog/newsroom.
  * *Market* — market-topic and regulatory-topic news queries, official government feeds chosen by geography (GOV.UK
    search Atom, US Federal Register RSS), GDELT global news, and feeds auto-discovered (`<link rel=alternate>`, `/feed`)
    on trade publications/regulators suggested in research.
  * *Internal* — uploads inbox (CSV, Excel, PDF, text).
* **Probe & justify:** every candidate is fetched once. `specific` = 0.5·share of records containing the company's problem-space
  terms + 3·mean TF-IDF similarity of the top-10 records to the company's interest document; `recent` = freshness of the newest
  record (≤7d → 1.0, ≤30d → 0.8, ≤90d → 0.5); `credible` = source-type/publisher prior (government 0.95, company's own page
  0.9, trade publication 0.8, news search 0.7, app reviews 0.6, Q&A 0.55). `overall = 0.45·specific + 0.30·recent +
  0.25·credible`. The reason text is written from the probe evidence (record counts, freshness, keyword share, an example
  headline). Sources ≥ 0.55 are auto-approved; the rest wait in the Source Manager.

## 4. Crawling & monitoring

* `fetch.py` checks robots.txt (incl. Crawl-delay) for every web host, rate-limits per host (2 s default; GDELT 5.5 s, Bing 3 s),
  backs off on 429/503, uses conditional GET (ETag/Last-Modified), and **never** attempts to pass CAPTCHAs, bot challenges or
  logins — such sources are recorded as blocked. Official APIs (Wikipedia, Stack Exchange, HN Algolia, GDELT, iTunes) are
  used under their published rate limits.
* `monitor.py` checks approved sources in parallel across hosts, logs a `source_checks` row per source (status, HTTP code,
  latency, found/new) and updates health counters. Records are de-duplicated by canonical-URL hash; near-duplicates (same
  content hash or token-set ratio ≥ 92) are folded into the original and counted as **corroboration** by another publisher.
  Web pages are snapshot and diffed — a change produces a record listing added/removed lines; the first visit is a baseline,
  never a "development".

## 5. Ranking & AI analysis

Per record (`analysis.py`), with **separate** measures:

| Measure | How |
|---|---|
| Strategic relevance | TF-IDF cosine vs the company interest document (profile + interview answers, weighted ×2) + boosts for mentioning the company, a tracked competitor, a served geography, problem-space terms |
| Business impact | topic weight (regulation 0.9, pricing 0.85, M&A 0.85, launch 0.8, funding 0.75 …) × (0.45 + 0.55·relevance) |
| Urgency | recency + urgency language ("effective from", "deadline", "ban") + regulation + strongly negative sentiment |
| Source reliability | publisher/source prior (government 0.95; recognised outlets ≥ 0.85; unknown publishers capped at 0.6) |
| Evidence strength | verification class (primary source 0.9, corroborated 0.8, internal 0.7, reported 0.6, user-generated 0.45) + corroboration |
| Recency / novelty | exp(−age/10 d); 1 − max similarity to earlier records |

`score = 0.30·relevance + 0.22·impact + 0.15·urgency + 0.11·reliability + 0.11·recency + 0.11·novelty`. Every record stores a
plain-language explanation of why it ranked where it did. The LLM's self-reported confidence is stored but **never** enters the
score. The top records are then interpreted by the LLM in batches (what happened — restricted to the evidence; material vs.
general news; why it matters for this company's offerings/segments/priorities; business area; opportunity/threat; urgency;
what to consider). Non-material items are dropped from "significant".

## 6. Monday Morning Brief

Two layers: an **evidence layer** selected deterministically (significant, de-duplicated records in the period, per category,
with links, dates, snippets and the three separate measures) and an **interpretation layer** (executive summary, opportunities,
risks, actions, watch-list). The interpretation prompt only sees numbered signals; any statement that does not cite a valid
signal ID is dropped, so the brief cannot contain uncited claims. A citation that exists is not yet a citation that *supports*
the claim, so a **grounding check** follows: a second, temperature-0 model pass receives each statement together with the verbatim
snippets it cites and answers *supported / partial / unsupported*. Unsupported statements are removed, partial ones are rewritten
to what the evidence says, and the brief header reports how many were checked, removed and corrected. Empty categories are stated explicitly with the number of
sources checked and records collected. Customer voice is aggregated from reviews (average rating, recurring complaint phrases,
verbatim quotes). Internal uploads are clustered (TF-IDF + KMeans) into themes; synthetic test files are labelled in the brief.
Outputs: dashboard view with action tracking, Markdown, printable HTML (→PDF), email-ready HTML (inline CSS), JSON, and
delivery via SMTP or an `.eml` in the outbox.

## 7. Scheduling, audit & reliability

* **A. Weekly brief** (cron, per-company day/time/time-zone) → crawl → analyse → brief → deliver.
  **B. Monthly audit** (cron) → dead URLs, repeated failures, inactive sources, duplicates (identical definitions or ≥ 50 %
  record overlap incl. cross-source near-duplicates), declining relevance (30-day vs prior average, or never significant),
  new candidates via discovery (left for approval), add/remove recommendations, source-health report.
  **Monitoring** runs at the configured interval between briefs. Manual triggers exist for all of them.
* Every run writes `job_runs` (status, progress log, traceback). Failed and **missed** runs are shown on every dashboard page;
  runs interrupted by a restart are marked failed on startup. A run can be missed in two ways and both are caught: the
  process was busy/paused (APScheduler `EVENT_JOB_MISSED`), or the process was not running at all (laptop asleep, free host
  idling) — on every start-up the scheduler recomputes each company's last weekly and monthly fire time, logs a `missed` row
  if no run covers it, and runs it immediately (catch-up).
* **External trigger for free hosts:** `POST /api/cron/{weekly_brief|monthly_audit|monitor}` with a bearer `CRON_TOKEN` runs the
  job for every configured company. `.github/workflows/scheduled-runs.yml` calls it on the same Monday/1st-of-month schedule,
  which also wakes an idled free instance.
* **Context change** (`context.py`): the founder describes what changed; the profile's monitoring fields are updated, every
  source is re-scored against old vs new interest documents, sources whose topic left the plan or whose relevance collapsed are
  retired, discovery adds new ones, and a revision report explains before → after.
* **LLM layer** (`llm.py`): free providers only — Groq free tier (primary), Gemini / OpenRouter free tiers, local Ollama,
  then a deterministic rule-based path. Free-tier quotas are *per model*, so the Groq provider is a chain of models (filtered
  to what the account can use via `GET /models`); a per-minute 429 waits for `retry-after`, a per-day quota or "request too
  large" parks that model and moves to the next one. Prompts carrying uploaded internal data go to the local model first so
  confidential CRM text stays on the machine when Ollama is available.
* Reliability details: per-model throttling, LLM response cache (re-runs are deterministic), JSON
  repair for truncated model output, WAL-mode SQLite with per-thread connections, idempotent inserts (unique URL hash).

## 8. Limitations (honest)

* News search relies on Bing News RSS and GDELT; GDELT rate-limits aggressively (429s are logged, the run continues).
* Reddit, Quora, Google Play reviews and most review sites forbid crawling — they are identified as customer venues but not
  scraped (Reddit can be enabled with free official API credentials).
* Local LLM quality (qwen2.5:3b on CPU — the demo laptop had ~4.6 GB free RAM, too little for 7B) is below frontier models; the evidence/interpretation separation and signal-ID
  citation checks are what keep the brief trustworthy. A free Groq/Gemini key can be dropped into `.env` for better prose.
* Founder interview answers in the demo were written by the candidate from public information (MeetUniversity's founder
  was not interviewed); they are labelled as such.
