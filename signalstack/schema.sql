-- Signal Stack database schema (SQLite).
-- JSON columns are stored as TEXT and parsed in the application layer.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    website       TEXT NOT NULL,
    domain        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'new',      -- new | researched | interviewing | stack_ready | monitoring
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- Structured company intelligence profile (one row per company, JSON document).
CREATE TABLE IF NOT EXISTS profiles (
    company_id    INTEGER PRIMARY KEY REFERENCES companies(id) ON DELETE CASCADE,
    data          TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL
);

-- Every profile change is versioned so context changes can be diffed.
CREATE TABLE IF NOT EXISTS profile_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,
    data          TEXT NOT NULL,
    reason        TEXT,
    created_at    TEXT NOT NULL
);

-- Pages read during initial research (evidence for facts).
CREATE TABLE IF NOT EXISTS research_pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    url           TEXT NOT NULL,
    kind          TEXT,                                -- homepage | about | services | news | wikipedia ...
    title         TEXT,
    text          TEXT,
    http_status   INTEGER,
    fetched_at    TEXT NOT NULL
);

-- Verified facts vs. assumptions produced by research.
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,                       -- fact | assumption
    field         TEXT,
    statement     TEXT NOT NULL,
    evidence      TEXT,
    source_url    TEXT,
    verification  TEXT,                                -- how it was verified / why it is only an assumption
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interview_questions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,                    -- order in the interview (follow-ups get fractional order via parent)
    parent_id     INTEGER REFERENCES interview_questions(id) ON DELETE CASCADE,
    area          TEXT NOT NULL,
    question      TEXT NOT NULL,
    rationale     TEXT,
    answer        TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',     -- pending | answered | skipped
    is_followup   INTEGER NOT NULL DEFAULT 0,
    generated_by  TEXT,
    asked_at      TEXT,
    answered_at   TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS competitors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    website       TEXT,
    domain        TEXT,
    description   TEXT,
    why           TEXT,
    kind          TEXT,                                -- direct | substitute | adjacent
    similarity    REAL,
    evidence_url  TEXT,
    evidence      TEXT,
    discovered_via TEXT,
    status        TEXT NOT NULL DEFAULT 'discovered',  -- discovered | approved | rejected
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id           INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    category             TEXT NOT NULL,                -- customers | competitors | market | internal
    type                 TEXT NOT NULL,                -- rss | bing_news | gdelt | webpage | hn | stackexchange | appstore_reviews | wikipedia | upload | reddit
    name                 TEXT NOT NULL,
    url                  TEXT,
    query                TEXT,
    params               TEXT,                         -- JSON adapter params
    competitor_id        INTEGER REFERENCES competitors(id) ON DELETE SET NULL,
    reason               TEXT NOT NULL,
    specific_score       REAL,
    recency_score        REAL,
    credibility_score    REAL,
    overall_score        REAL,
    probe                TEXT,                         -- JSON: probe evidence collected at discovery time
    status               TEXT NOT NULL DEFAULT 'discovered', -- discovered | approved | rejected | removed | blocked
    origin               TEXT NOT NULL DEFAULT 'discovery',  -- discovery | user | audit | context_change
    duplicate_of         INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at           TEXT NOT NULL,
    approved_at          TEXT,
    removed_at           TEXT,
    last_checked         TEXT,
    last_success         TEXT,
    last_new_item_at     TEXT,
    last_status          TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    items_total          INTEGER NOT NULL DEFAULT 0,
    etag                 TEXT,
    last_modified        TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_company ON sources(company_id, status);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id      INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    trigger         TEXT NOT NULL,                     -- manual | weekly_brief | monitor | demo
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    sources_checked INTEGER DEFAULT 0,
    items_found     INTEGER DEFAULT 0,
    items_new       INTEGER DEFAULT 0,
    items_duplicate INTEGER DEFAULT 0,
    errors          TEXT
);

CREATE TABLE IF NOT EXISTS source_checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES crawl_runs(id) ON DELETE CASCADE,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    checked_at    TEXT NOT NULL,
    status        TEXT NOT NULL,                       -- ok | not_modified | error | robots_blocked | needs_credentials
    http_status   INTEGER,
    latency_ms    INTEGER,
    items_found   INTEGER DEFAULT 0,
    items_new     INTEGER DEFAULT 0,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_checks_source ON source_checks(source_id, checked_at);

-- Snapshots for change detection on watched web pages.
CREATE TABLE IF NOT EXISTS page_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    fetched_at    TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    text          TEXT
);

-- Collected intelligence records.
CREATE TABLE IF NOT EXISTS items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id        INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    source_id         INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    run_id            INTEGER REFERENCES crawl_runs(id) ON DELETE SET NULL,
    category          TEXT NOT NULL,
    competitor_id     INTEGER REFERENCES competitors(id) ON DELETE SET NULL,
    url               TEXT,
    url_hash          TEXT NOT NULL,
    title             TEXT NOT NULL,
    snippet           TEXT,
    publisher         TEXT,
    published_at      TEXT,
    discovered_at     TEXT NOT NULL,
    content_hash      TEXT,
    dup_of            INTEGER REFERENCES items(id) ON DELETE SET NULL,
    corroborations    INTEGER NOT NULL DEFAULT 0,
    -- analysis -------------------------------------------------------------
    topic             TEXT,
    sentiment         REAL,
    relevance         REAL,     -- strategic relevance to this company (0-1)
    impact            REAL,
    urgency           REAL,
    credibility       REAL,     -- source reliability (0-1), independent of the AI
    recency           REAL,
    novelty           REAL,
    evidence_strength REAL,     -- how well the claim is evidenced (primary source, corroboration)
    score             REAL,
    polarity          TEXT,     -- opportunity | threat | neutral
    verification      TEXT,     -- primary_source | reported | user_generated | internal | unverified
    significant       INTEGER NOT NULL DEFAULT 0,
    explanation       TEXT,
    analysis          TEXT,     -- JSON (what/when/why/area/action...) - interpretation, labelled as such
    ai_confidence     REAL,     -- the model's self-reported confidence: NOT evidence
    analysed_by       TEXT,
    analysed_at       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_items_unique ON items(company_id, url_hash);
CREATE INDEX IF NOT EXISTS idx_items_company ON items(company_id, discovered_at);

CREATE TABLE IF NOT EXISTS uploads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    source_id     INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    filename      TEXT NOT NULL,
    kind          TEXT NOT NULL,                        -- csv | xlsx | pdf | txt
    label         TEXT,                                 -- crm_export | call_transcripts | support | feedback | other
    is_synthetic  INTEGER NOT NULL DEFAULT 0,
    records       INTEGER,
    summary       TEXT,                                 -- JSON
    uploaded_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS briefs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    period_start  TEXT,
    period_end    TEXT,
    trigger       TEXT NOT NULL,
    run_id        INTEGER REFERENCES crawl_runs(id) ON DELETE SET NULL,
    title         TEXT,
    markdown      TEXT NOT NULL,
    html          TEXT NOT NULL,
    data          TEXT,                                 -- JSON structured brief
    llm           TEXT,
    delivered_at  TEXT,
    delivery      TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id      INTEGER NOT NULL REFERENCES briefs(id) ON DELETE CASCADE,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    priority      TEXT,
    action        TEXT NOT NULL,
    reason        TEXT,
    owner         TEXT,
    signal_ids    TEXT,
    status        TEXT NOT NULL DEFAULT 'open',          -- open | in_progress | done | dismissed
    note          TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS audits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    trigger       TEXT NOT NULL,
    report        TEXT NOT NULL,                        -- JSON
    markdown      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stack_revisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    reason        TEXT,
    diff          TEXT,                                 -- JSON
    markdown      TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    company_id              INTEGER PRIMARY KEY REFERENCES companies(id) ON DELETE CASCADE,
    brief_enabled           INTEGER NOT NULL DEFAULT 1,
    briefing_day            TEXT NOT NULL DEFAULT 'mon',
    briefing_time           TEXT NOT NULL DEFAULT '07:30',
    timezone                TEXT NOT NULL DEFAULT 'Asia/Kolkata',
    monitor_enabled         INTEGER NOT NULL DEFAULT 1,
    monitor_every_hours     INTEGER NOT NULL DEFAULT 24,
    source_refresh_days     INTEGER NOT NULL DEFAULT 7,
    audit_enabled           INTEGER NOT NULL DEFAULT 1,
    audit_day_of_month      INTEGER NOT NULL DEFAULT 1,
    audit_time              TEXT NOT NULL DEFAULT '06:00',
    email_to                TEXT,
    updated_at              TEXT
);

-- Scheduled/background job log. Failures are surfaced in the dashboard.
CREATE TABLE IF NOT EXISTS job_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    job           TEXT NOT NULL,                        -- weekly_brief | monthly_audit | monitor | research | discovery | manual_brief ...
    trigger       TEXT NOT NULL,                        -- schedule | manual | api
    scheduled_for TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',      -- running | success | failed | missed
    progress      TEXT,
    log           TEXT,
    error         TEXT,
    result_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON job_runs(company_id, started_at);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    purpose       TEXT,
    provider      TEXT,
    model         TEXT,
    prompt_hash   TEXT,
    ok            INTEGER,
    latency_ms    INTEGER,
    error         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash   TEXT PRIMARY KEY,
    provider      TEXT,
    response      TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
