# Cost statement

**Total cost to build, run and demonstrate: ₹0.** No paid API, paid scraping platform or paid AI service is used.

| Component | What it is | Cost / licence | Limitations disclosed |
|---|---|---|---|
| Python 3.12, FastAPI, Uvicorn, Jinja2 | Web app & API | Open source (PSF, MIT, BSD) | — |
| SQLite | Database | Public domain | Single-node; fine for one team, swap for Postgres to scale |
| APScheduler | Weekly brief / monthly audit / monitoring schedules | MIT | Runs inside the app process: the process must be running at the scheduled time (missed runs are logged) |
| scikit-learn, pandas, VADER, rapidfuzz, feedparser, BeautifulSoup, lxml, pypdf, openpyxl | NLP, ranking, parsing | BSD / MIT | VADER is tuned for English social text |
| Ollama + Qwen2.5 (optional local) | Local open-weights model; keeps internal data on the machine | Free (Qwen2.5-7B Apache-2.0, 3B Qwen Research licence) | CPU-only is slow (minutes per long prompt) and a 3B model is much weaker; used as a fallback |
| Google Gemini / OpenRouter `:free` (optional) | Alternative free-tier hosted LLMs | Free tier, no card | Rate limits; not used in the demo |
| **Groq free tier** (demo LLM: `openai/gpt-oss-120b`, falling back across other free models) | Hosted open-weights models for interpretation | Free tier, no card | Per-model limits (requests/min, tokens/min, tokens/day); the app rotates across models and waits on 429s. Prompts go to Groq, so internal uploads are routed to the local model first when Ollama is running |
| Pollinations text API (optional fallback) | Keyless public LLM endpoint | Free | 1 reply / 45 s per IP, ~800 output tokens — last-resort fallback only |
| Rule-based fallback | Deterministic analysis when no LLM is reachable | Free | Plainer prose; no competitor suggestions without an LLM or founder input |
| Bing News RSS | News search per query | Free public feed (robots.txt allows `/news/`) | ~10–25 most recent items per query; no full text |
| GDELT DOC 2.0 API | Global news index | Free | ~1 request / 5 s; frequently returns HTTP 429 — logged and skipped |
| Stack Exchange API | Real customer questions | Free (300 requests/day/IP without a key) | Daily quota |
| Hacker News Algolia API | Practitioner discussion | Free | — |
| Apple iTunes Search API + customer-review RSS | App discovery and real customer reviews | Free | Most recent ~50 reviews per app and country |
| Wikipedia Action API | Company background | Free | Only notable companies have articles |
| GOV.UK search Atom, US Federal Register RSS, publisher RSS feeds | Primary-source regulatory/industry news | Free | Coverage depends on the geography/publisher |
| Reddit API (optional) | Customer communities | Free official API credentials | Not crawled without credentials (Reddit's robots.txt forbids crawling) |
| Render free web service (deployment) | Hosts the dashboard + scheduler | Free plan, no card | Sleeps after ~15 min idle (first request ~1 min); ephemeral disk, so the demo DB is re-seeded on each deploy |
| GitHub Actions (external cron) | Wakes the free host Monday / 1st of month and calls `/api/cron/*` | Free for this volume | Scheduled workflows can start a few minutes late |
| SMTP (optional) | Brief delivery | Any free mailbox | Without SMTP the brief is written as an `.eml` to `data/outbox/` |

Not used on purpose: Google Search/News scraping (robots.txt disallows), DuckDuckGo HTML (returns a bot challenge — never bypassed),
Reddit/Quora/Google Play scraping (terms forbid), any paid SERP or scraping API.
