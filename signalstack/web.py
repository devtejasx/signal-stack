"""FastAPI dashboard + JSON API."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import markdown as mdlib
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import adapters, analysis, brief, config, context, db, interview, llm, pipeline, scheduler
from .internal import ingest_upload
from .jobs import is_running, mark_stale_runs, run_in_background
from .profile import FIELD_LABEL, INTEL_FIELDS, get_profile, save_profile
from .textutil import domain_of, truncate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    mark_stale_runs()
    if config.ENABLE_SCHEDULER:
        scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="Signal Stack", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
templates.env.filters["trunc"] = lambda s, n=120: truncate(str(s or ""), n)
templates.env.filters["md"] = lambda s: mdlib.markdown(s or "", extensions=["tables"])
templates.env.filters["loads"] = lambda s: db.loads(s, {}) if isinstance(s, str) else (s or {})
templates.env.filters["dt"] = lambda s: (s or "")[:16].replace("T", " ")
templates.env.filters["pct"] = lambda v: "—" if v is None else f"{v:.2f}"
templates.env.globals["FIELD_LABEL"] = FIELD_LABEL


def company_or_404(cid: int) -> dict:
    c = db.one("SELECT * FROM companies WHERE id=?", (cid,))
    if not c:
        raise HTTPException(404, "company not found")
    return c


def render(request: Request, name: str, **ctx) -> HTMLResponse:
    cid = ctx.get("company", {}).get("id") if isinstance(ctx.get("company"), dict) else None
    failures = []
    if cid:
        failures = db.q("SELECT * FROM job_runs WHERE company_id=? AND status IN ('failed','missed') AND "
                        "started_at >= datetime('now','-14 days') ORDER BY id DESC LIMIT 3", (cid,))
        ctx.setdefault("running", db.q("SELECT * FROM job_runs WHERE company_id=? AND status='running' ORDER BY id DESC", (cid,)))
    ctx.update(request=request, llm_label=llm.active_provider_label(), recent_failures=failures,
               scheduler_on=scheduler.running(), companies=db.q("SELECT id, name FROM companies ORDER BY id"))
    return templates.TemplateResponse(request, name, ctx)


def back(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


# --------------------------------------------------------------------------- companies
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    rows = db.q("""SELECT c.*, (SELECT COUNT(*) FROM sources s WHERE s.company_id=c.id AND s.status='approved') sources,
                   (SELECT COUNT(*) FROM items i WHERE i.company_id=c.id) items,
                   (SELECT COUNT(*) FROM briefs b WHERE b.company_id=c.id) briefs,
                   (SELECT COUNT(*) FROM competitors k WHERE k.company_id=c.id AND k.status='approved') competitors
                   FROM companies c ORDER BY c.id""")
    return render(request, "home.html", rows=rows)


@app.post("/companies")
def add_company(name: str = Form(...), website: str = Form(...)):
    cid = pipeline.create_company(name, website)
    if not db.one("SELECT 1 FROM research_pages WHERE company_id=?", (cid,)):
        run_in_background(cid, "research", "manual", pipeline.research_job(cid))
    return back(f"/c/{cid}")


@app.get("/c/{cid}", response_class=HTMLResponse)
def overview(request: Request, cid: int):
    c = company_or_404(cid)
    p = get_profile(cid)
    stats = {
        "facts": db.scalar("SELECT COUNT(*) FROM facts WHERE company_id=? AND kind='fact'", (cid,)),
        "assumptions": db.scalar("SELECT COUNT(*) FROM facts WHERE company_id=? AND kind='assumption'", (cid,)),
        "interview": interview.progress(cid),
        "sources": {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) n FROM sources WHERE company_id=? GROUP BY status", (cid,))},
        "by_cat": {r["category"]: r["n"] for r in db.q("SELECT category, COUNT(*) n FROM sources WHERE company_id=? AND status='approved' GROUP BY category", (cid,))},
        "competitors": db.scalar("SELECT COUNT(*) FROM competitors WHERE company_id=? AND status='approved'", (cid,)),
        "records": db.scalar("SELECT COUNT(*) FROM items WHERE company_id=?", (cid,)),
        "significant": db.scalar("SELECT COUNT(*) FROM items WHERE company_id=? AND significant=1", (cid,)),
        "crawls": db.scalar("SELECT COUNT(*) FROM crawl_runs WHERE company_id=?", (cid,)),
        "briefs": db.scalar("SELECT COUNT(*) FROM briefs WHERE company_id=?", (cid,)),
        "audits": db.scalar("SELECT COUNT(*) FROM audits WHERE company_id=?", (cid,)),
    }
    latest = db.one("SELECT id, title, created_at FROM briefs WHERE company_id=? ORDER BY id DESC LIMIT 1", (cid,))
    top = db.q("SELECT * FROM items WHERE company_id=? AND significant=1 AND dup_of IS NULL ORDER BY score DESC LIMIT 6", (cid,))
    jobs = db.q("SELECT * FROM job_runs WHERE company_id=? ORDER BY id DESC LIMIT 8", (cid,))
    return render(request, "overview.html", company=c, profile=p, stats=stats, latest=latest, top=top, jobs=jobs,
                  next_runs=scheduler.next_runs(cid))


@app.post("/c/{cid}/run/{job}")
def run_step(cid: int, job: str):
    company_or_404(cid)
    makers = {"research": pipeline.research_job(cid), "discovery": pipeline.discovery_job(cid),
              "monitor": pipeline.monitor_job(cid, "manual"), "manual_brief": pipeline.weekly_brief_job(cid, "manual"),
              "monthly_audit": pipeline.audit_job(cid, "manual")}
    if job not in makers:
        raise HTTPException(400, "unknown job")
    run_in_background(cid, job, "manual", makers[job])
    scheduler.sync_jobs()
    return back(f"/c/{cid}/runs" if job != "research" else f"/c/{cid}")


@app.api_route("/api/cron/{job}", methods=["GET", "POST"])
def cron_trigger(job: str, request: Request):
    """External-scheduler hook. Free hosts idle the process, so an outside cron (GitHub Actions) can wake it and run
    the two automated processes for every configured company. Requires the CRON_TOKEN bearer secret."""
    if not config.CRON_TOKEN:
        raise HTTPException(404, "cron endpoint disabled (set CRON_TOKEN)")
    import hmac
    sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(sent, config.CRON_TOKEN):
        raise HTTPException(401, "bad token")
    makers = {"weekly_brief": pipeline.weekly_brief_job, "monthly_audit": pipeline.audit_job, "monitor": pipeline.monitor_job}
    if job not in makers:
        raise HTTPException(400, "unknown job")
    started = []
    for c in db.q("SELECT id FROM companies WHERE status IN ('stack_ready','monitoring') ORDER BY id"):
        started.append({"company_id": c["id"], "job_id": run_in_background(c["id"], job, "external_cron", makers[job](c["id"], "external_cron"))})
    return {"job": job, "started": started}


@app.get("/api/jobs/{jid}")
def job_status(jid: int):
    j = db.one("SELECT id, job, status, progress, error, started_at, finished_at FROM job_runs WHERE id=?", (jid,))
    if not j:
        raise HTTPException(404)
    return j


# --------------------------------------------------------------------------- profile
@app.get("/c/{cid}/profile", response_class=HTMLResponse)
def profile_page(request: Request, cid: int):
    c = company_or_404(cid)
    p = get_profile(cid)
    facts = db.q("SELECT * FROM facts WHERE company_id=? ORDER BY kind DESC, id", (cid,))
    qa = db.q("SELECT * FROM interview_questions WHERE company_id=? ORDER BY seq, is_followup, id", (cid,))
    revisions = db.q("SELECT * FROM stack_revisions WHERE company_id=? ORDER BY id DESC", (cid,))
    history = db.q("SELECT version, reason, created_at FROM profile_history WHERE company_id=? ORDER BY id DESC LIMIT 25", (cid,))
    pages = db.q("SELECT url, kind, title, fetched_at FROM research_pages WHERE company_id=?", (cid,))
    return render(request, "profile.html", company=c, p=p, facts=facts, qa=qa, revisions=revisions, history=history,
                  pages=pages, intel_fields=INTEL_FIELDS)


LIST_KEYS = ["offerings", "customer_segments", "geographies", "value_props", "revenue_streams", "category_keywords",
             "customer_problem_queries", "market_topics", "regulatory_topics"]


@app.post("/c/{cid}/profile")
async def profile_save(request: Request, cid: int):
    company_or_404(cid)
    form = await request.form()
    p = get_profile(cid)
    for k in ("summary", "industry", "business_model"):
        if k in form:
            p[k] = str(form[k]).strip()
    for k in LIST_KEYS:
        if k in form:
            p[k] = [l.strip() for l in str(form[k]).splitlines() if l.strip()]
    for k, _ in INTEL_FIELDS:
        fk = f"intel_{k}"
        if fk in form:
            lines = [l.strip() for l in str(form[fk]).splitlines() if l.strip()]
            old = {(v.get("text") if isinstance(v, dict) else str(v)): v for v in p["intelligence"].get(k, [])}
            p["intelligence"][k] = [old.get(l) or {"text": l, "source": "edited"} for l in lines]
    save_profile(cid, p, "edited by founder in dashboard")
    analysis.score_items(cid)
    return back(f"/c/{cid}/profile?saved=1")


@app.post("/c/{cid}/context")
def context_change(cid: int, change: str = Form(...)):
    company_or_404(cid)
    run_in_background(cid, "context_change", "manual", lambda ctx: f"revision={context.apply_context_change(cid, change, ctx)}")
    return back(f"/c/{cid}/runs")


@app.get("/c/{cid}/revisions/{rid}", response_class=HTMLResponse)
def revision(request: Request, cid: int, rid: int):
    c = company_or_404(cid)
    r = db.one("SELECT * FROM stack_revisions WHERE id=? AND company_id=?", (rid, cid))
    return render(request, "doc.html", company=c, title="Signal Stack revision", body=r["markdown"])


# --------------------------------------------------------------------------- interview
@app.get("/c/{cid}/interview", response_class=HTMLResponse)
def interview_page(request: Request, cid: int):
    c = company_or_404(cid)
    if not db.one("SELECT 1 FROM research_pages WHERE company_id=?", (cid,)):
        return render(request, "interview.html", company=c, needs_research=True, current=None, done=[], progress={})
    if not db.one("SELECT 1 FROM interview_questions WHERE company_id=?", (cid,)):
        interview.plan_interview(cid)
    current = interview.next_question(cid)
    done = [q for q in interview.ordered(cid) if q["status"] != "pending"]
    return render(request, "interview.html", company=c, current=current, done=done, progress=interview.progress(cid),
                  needs_research=False)


@app.post("/c/{cid}/interview/{qid}/answer")
def interview_answer(cid: int, qid: int, answer: str = Form(...)):
    if answer.strip():
        interview.answer_question(cid, qid, answer)
    interview.finish_if_complete(cid)
    return back(f"/c/{cid}/interview#current")


@app.post("/c/{cid}/interview/{qid}/skip")
def interview_skip(cid: int, qid: int):
    interview.skip_question(cid, qid)
    interview.finish_if_complete(cid)
    return back(f"/c/{cid}/interview#current")


@app.post("/c/{cid}/interview/{qid}/edit")
def interview_edit(cid: int, qid: int, answer: str = Form("")):
    interview.edit_answer(cid, qid, answer)
    analysis.score_items(cid)
    return back(f"/c/{cid}/profile#interview")


@app.post("/c/{cid}/interview/reset")
def interview_reset(cid: int):
    db.execute("DELETE FROM interview_questions WHERE company_id=?", (cid,))
    return back(f"/c/{cid}/interview")


# --------------------------------------------------------------------------- signal stack & sources
@app.get("/c/{cid}/stack", response_class=HTMLResponse)
def stack_page(request: Request, cid: int):
    c = company_or_404(cid)
    sources = db.q("SELECT s.*, k.name AS competitor FROM sources s LEFT JOIN competitors k ON k.id=s.competitor_id "
                   "WHERE s.company_id=? AND s.status='approved' ORDER BY s.category, s.overall_score DESC", (cid,))
    comps = db.q("SELECT * FROM competitors WHERE company_id=? ORDER BY CASE status WHEN 'approved' THEN 0 WHEN 'discovered' THEN 1 "
                 "ELSE 2 END, similarity DESC", (cid,))
    counts = {r["competitor_id"]: r["n"] for r in db.q("SELECT competitor_id, COUNT(*) n FROM items WHERE company_id=? "
                                                         "AND competitor_id IS NOT NULL GROUP BY competitor_id", (cid,))}
    return render(request, "stack.html", company=c, sources=sources, comps=comps, comp_counts=counts)


@app.post("/c/{cid}/competitors/{kid}/{action}")
def competitor_action(cid: int, kid: int, action: str):
    status = {"approve": "approved", "reject": "rejected"}.get(action)
    if not status:
        raise HTTPException(400)
    db.execute("UPDATE competitors SET status=? WHERE id=? AND company_id=?", (status, kid, cid))
    if status == "rejected":
        db.execute("UPDATE sources SET status='removed', removed_at=? WHERE competitor_id=? AND status IN ('approved','discovered')",
                   (db.now(), kid))
    return back(f"/c/{cid}/stack#competitors")


@app.post("/c/{cid}/competitors")
def competitor_add(cid: int, name: str = Form(...), website: str = Form(...), why: str = Form("")):
    company_or_404(cid)
    p = get_profile(cid)
    p.setdefault("competitor_candidates", []).append({"name": name, "domain": domain_of(website), "why": why or "Added by founder", "kind": "direct"})
    p.setdefault("founder_named", {}).setdefault("competitors", []).append(name)
    save_profile(cid, p, f"competitor added: {name}")
    run_in_background(cid, "discovery", "manual", pipeline.discovery_job(cid))
    return back(f"/c/{cid}/runs")


@app.get("/c/{cid}/sources", response_class=HTMLResponse)
def sources_page(request: Request, cid: int, status: str = "", category: str = ""):
    c = company_or_404(cid)
    sql = ("SELECT s.*, k.name AS competitor, d.name AS duplicate_name FROM sources s LEFT JOIN competitors k ON k.id=s.competitor_id "
           "LEFT JOIN sources d ON d.id=s.duplicate_of WHERE s.company_id=?")
    params: list = [cid]
    if status:
        sql += " AND s.status=?"
        params.append(status)
    if category:
        sql += " AND s.category=?"
        params.append(category)
    sql += " ORDER BY CASE s.status WHEN 'discovered' THEN 0 WHEN 'approved' THEN 1 WHEN 'blocked' THEN 2 ELSE 3 END, s.category, s.overall_score DESC"
    rows = db.q(sql, params)
    counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) n FROM sources WHERE company_id=? GROUP BY status", (cid,))}
    uploads = db.q("SELECT * FROM uploads WHERE company_id=? ORDER BY id DESC", (cid,))
    comps = db.q("SELECT id, name FROM competitors WHERE company_id=? AND status='approved'", (cid,))
    return render(request, "sources.html", company=c, rows=rows, counts=counts, status=status, category=category,
                  uploads=uploads, comps=comps)


@app.post("/c/{cid}/sources/{sid}/{action}")
def source_action(cid: int, sid: int, action: str):
    mapping = {"approve": "approved", "reject": "rejected", "remove": "removed", "restore": "approved"}
    if action not in mapping:
        raise HTTPException(400)
    fields = {"status": mapping[action]}
    if mapping[action] == "approved":
        fields["approved_at"] = db.now()
    if mapping[action] == "removed":
        fields["removed_at"] = db.now()
    db.execute(f"UPDATE sources SET {', '.join(k + '=?' for k in fields)} WHERE id=? AND company_id=?", (*fields.values(), sid, cid))
    return back(f"/c/{cid}/sources")


@app.post("/c/{cid}/sources")
def source_add(cid: int, category: str = Form(...), type: str = Form(...), name: str = Form(...), url: str = Form(""),
               query: str = Form(""), reason: str = Form(...), competitor_id: str = Form("")):
    company_or_404(cid)
    if type == "bing_news" and query and not url:
        url = adapters.bing_news_url(query)
    if type == "rss" and url:
        found = adapters.discover_feed(url)
        if found:
            url = found
    from .discovery import credibility_for
    db.insert("sources", {"company_id": cid, "category": category, "type": type, "name": name, "url": url or None,
                          "query": query or None, "params": {}, "competitor_id": int(competitor_id) if competitor_id else None,
                          "reason": reason, "credibility_score": credibility_for(type, url), "status": "approved",
                          "origin": "user", "created_at": db.now(), "approved_at": db.now()})
    return back(f"/c/{cid}/sources")


@app.get("/c/{cid}/sources/{sid}", response_class=HTMLResponse)
def source_detail(request: Request, cid: int, sid: int):
    c = company_or_404(cid)
    s = db.one("SELECT * FROM sources WHERE id=? AND company_id=?", (sid, cid))
    if not s:
        raise HTTPException(404)
    checks = db.q("SELECT * FROM source_checks WHERE source_id=? ORDER BY id DESC LIMIT 50", (sid,))
    items = db.q("SELECT * FROM items WHERE source_id=? ORDER BY id DESC LIMIT 50", (sid,))
    snaps = db.q("SELECT id, fetched_at, content_hash, length(text) len FROM page_snapshots WHERE source_id=? ORDER BY id DESC LIMIT 20", (sid,))
    dups = db.q("SELECT id, name FROM sources WHERE duplicate_of=?", (sid,))
    return render(request, "source_detail.html", company=c, s=s, checks=checks, items=items, snaps=snaps, dups=dups,
                  probe=db.loads(s["probe"], {}))


@app.post("/c/{cid}/upload")
async def upload(cid: int, file: UploadFile = File(...), label: str = Form("other"), synthetic: str = Form("")):
    company_or_404(cid)
    data = await file.read()
    try:
        ingest_upload(cid, file.filename or "upload.txt", data, label, bool(synthetic))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    analysis.score_items(cid)
    analysis.interpret_items(cid, limit=12)
    return back(f"/c/{cid}/sources?category=internal")


# --------------------------------------------------------------------------- feed
@app.get("/c/{cid}/feed", response_class=HTMLResponse)
def feed(request: Request, cid: int, category: str = "", competitor: str = "", days: int = 30, min_rel: float = 0.0,
         polarity: str = "", verification: str = "", significant: str = "", impact: str = "", sort: str = "date", q: str = ""):
    c = company_or_404(cid)
    sql = ("SELECT i.*, s.name AS source_name, k.name AS competitor FROM items i LEFT JOIN sources s ON s.id=i.source_id "
           "LEFT JOIN competitors k ON k.id=i.competitor_id WHERE i.company_id=? AND i.dup_of IS NULL "
           "AND COALESCE(i.published_at, i.discovered_at) >= datetime('now', ?)")
    params: list = [cid, f"-{int(days)} days"]
    for col, val in (("i.category", category), ("i.polarity", polarity), ("i.verification", verification)):
        if val:
            sql += f" AND {col}=?"
            params.append(val)
    if competitor:
        sql += " AND i.competitor_id=?"
        params.append(int(competitor))
    if min_rel:
        sql += " AND i.relevance>=?"
        params.append(min_rel)
    if impact:
        sql += " AND i.impact>=?"
        params.append({"high": 0.6, "medium": 0.4}.get(impact, 0))
    if significant:
        sql += " AND i.significant=1"
    if q:
        sql += " AND (i.title LIKE ? OR i.snippet LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY " + ("i.score DESC" if sort == "score" else "COALESCE(i.published_at, i.discovered_at) DESC") + " LIMIT 300"
    rows = db.q(sql, params)
    comps = db.q("SELECT id, name FROM competitors WHERE company_id=? AND status='approved' ORDER BY name", (cid,))
    f = dict(category=category, competitor=competitor, days=days, min_rel=min_rel, polarity=polarity, verification=verification,
             significant=significant, impact=impact, sort=sort, q=q)
    return render(request, "feed.html", company=c, rows=rows, comps=comps, f=f)


@app.get("/c/{cid}/items/{iid}", response_class=HTMLResponse)
def item_detail(request: Request, cid: int, iid: int):
    c = company_or_404(cid)
    it = db.one("SELECT i.*, s.name AS source_name, s.reason AS source_reason, s.type AS source_type FROM items i "
                "LEFT JOIN sources s ON s.id=i.source_id WHERE i.id=? AND i.company_id=?", (iid, cid))
    if not it:
        raise HTTPException(404)
    dups = db.q("SELECT id, title, publisher, url FROM items WHERE dup_of=?", (iid,))
    return render(request, "item.html", company=c, it=it, a=db.loads(it["analysis"], {}), dups=dups)


# --------------------------------------------------------------------------- briefs
@app.get("/c/{cid}/briefs", response_class=HTMLResponse)
def briefs_page(request: Request, cid: int):
    c = company_or_404(cid)
    rows = db.q("SELECT id, title, created_at, trigger, llm, period_start, period_end, delivery FROM briefs WHERE company_id=? ORDER BY id DESC", (cid,))
    return render(request, "briefs.html", company=c, rows=rows)


@app.get("/c/{cid}/briefs/{bid}", response_class=HTMLResponse)
def brief_page(request: Request, cid: int, bid: int):
    c = company_or_404(cid)
    b = db.one("SELECT * FROM briefs WHERE id=? AND company_id=?", (bid, cid))
    if not b:
        raise HTTPException(404)
    acts = db.q("SELECT * FROM actions WHERE brief_id=? ORDER BY priority, id", (bid,))
    body = mdlib.markdown(b["markdown"], extensions=["tables"])
    return render(request, "brief.html", company=c, b=b, body=body, acts=acts)


@app.get("/c/{cid}/briefs/{bid}/export.{fmt}")
def brief_export(cid: int, bid: int, fmt: str):
    b = db.one("SELECT * FROM briefs WHERE id=? AND company_id=?", (bid, cid))
    if not b:
        raise HTTPException(404)
    fname = f"monday-brief-{cid}-{b['created_at'][:10]}"
    if fmt == "md":
        return PlainTextResponse(b["markdown"], headers={"Content-Disposition": f"attachment; filename={fname}.md"})
    if fmt == "html":
        return HTMLResponse(b["html"])
    if fmt == "email":
        return HTMLResponse(brief.email_html(b))
    if fmt == "json":
        return JSONResponse(db.loads(b["data"], {}))
    raise HTTPException(400)


@app.post("/c/{cid}/actions/{aid}")
def action_update(cid: int, aid: int, status: str = Form(...), note: str = Form("")):
    db.execute("UPDATE actions SET status=?, note=?, updated_at=? WHERE id=? AND company_id=?", (status, note, db.now(), aid, cid))
    a = db.one("SELECT brief_id FROM actions WHERE id=?", (aid,))
    return back(f"/c/{cid}/briefs/{a['brief_id']}#actions")


@app.post("/c/{cid}/briefs/{bid}/deliver")
def brief_deliver(cid: int, bid: int):
    pipeline.deliver(cid, bid)
    return back(f"/c/{cid}/briefs/{bid}")


# --------------------------------------------------------------------------- audits, runs, settings
@app.get("/c/{cid}/audits", response_class=HTMLResponse)
def audits_page(request: Request, cid: int, aid: int | None = None):
    c = company_or_404(cid)
    rows = db.q("SELECT id, created_at, trigger FROM audits WHERE company_id=? ORDER BY id DESC", (cid,))
    cur = db.one("SELECT * FROM audits WHERE id=? AND company_id=?", (aid, cid)) if aid else \
        db.one("SELECT * FROM audits WHERE company_id=? ORDER BY id DESC LIMIT 1", (cid,))
    return render(request, "audits.html", company=c, rows=rows, cur=cur, report=db.loads(cur["report"], {}) if cur else {})


@app.get("/c/{cid}/runs", response_class=HTMLResponse)
def runs_page(request: Request, cid: int):
    c = company_or_404(cid)
    jobs = db.q("SELECT * FROM job_runs WHERE company_id=? ORDER BY id DESC LIMIT 100", (cid,))
    crawls = db.q("SELECT * FROM crawl_runs WHERE company_id=? ORDER BY id DESC LIMIT 50", (cid,))
    return render(request, "runs.html", company=c, jobs=jobs, crawls=crawls, next_runs=scheduler.next_runs(cid))


@app.get("/c/{cid}/settings", response_class=HTMLResponse)
def settings_page(request: Request, cid: int):
    c = company_or_404(cid)
    s = db.one("SELECT * FROM settings WHERE company_id=?", (cid,))
    return render(request, "settings.html", company=c, s=s, next_runs=scheduler.next_runs(cid))


@app.post("/c/{cid}/settings")
async def settings_save(request: Request, cid: int):
    company_or_404(cid)
    form = await request.form()
    fields = {}
    for k in ("briefing_day", "briefing_time", "timezone", "audit_time", "email_to"):
        if k in form:
            fields[k] = str(form[k]).strip()
    for k in ("monitor_every_hours", "source_refresh_days", "audit_day_of_month"):
        if k in form and str(form[k]).strip():
            fields[k] = int(form[k])
    for k in ("brief_enabled", "monitor_enabled", "audit_enabled"):
        fields[k] = 1 if form.get(k) else 0
    fields["updated_at"] = db.now()
    db.update("settings", cid, fields, key="company_id")
    scheduler.sync_jobs()
    return back(f"/c/{cid}/settings?saved=1")


@app.get("/health")
def health():
    return {"ok": True, "llm": llm.active_provider_label(), "scheduler": scheduler.running()}
