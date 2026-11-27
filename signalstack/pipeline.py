"""High-level pipelines used by the dashboard, the scheduler and the CLI."""
from __future__ import annotations

import smtplib
from email.message import EmailMessage

from . import analysis, audit, brief, config, db, discovery, monitor, research
from .jobs import JobContext
from .textutil import domain_of, normalise_website


def create_company(name: str, website: str) -> int:
    website = normalise_website(website)
    existing = db.one("SELECT id FROM companies WHERE domain=?", (domain_of(website),))
    if existing:
        return existing["id"]
    cid = db.insert("companies", {"name": name.strip(), "website": website, "domain": domain_of(website), "status": "new",
                                  "created_at": db.now(), "updated_at": db.now()})
    db.insert("settings", {"company_id": cid, "updated_at": db.now()})
    return cid


def research_job(company_id: int):
    return lambda ctx: research.research_company(company_id, ctx) and "profile ready"


def discovery_job(company_id: int):
    return lambda ctx: discovery.build_signal_stack(company_id, ctx)


def monitor_job(company_id: int, trigger: str = "monitor"):
    def fn(ctx: JobContext):
        run_id = monitor.run_crawl(company_id, trigger, ctx)
        analysis.analyse(company_id, ctx)
        return f"crawl_run={run_id}"
    return fn


def weekly_brief_job(company_id: int, trigger: str = "weekly_brief"):
    """A. Weekly Intelligence Brief: check sources -> collect -> detect changes -> analyse -> report -> deliver."""
    def fn(ctx: JobContext):
        run_id = monitor.run_crawl(company_id, trigger, ctx)
        analysis.analyse(company_id, ctx)
        bid = brief.generate_brief(company_id, trigger=trigger, run_id=run_id, ctx=ctx)
        deliver(company_id, bid, ctx)
        return f"brief={bid}"
    return fn


def audit_job(company_id: int, trigger: str = "monthly_audit"):
    """B. Monthly Signal Stack Audit."""
    return lambda ctx: f"audit={audit.run_audit(company_id, trigger, ctx)}"


def deliver(company_id: int, brief_id: int, ctx: JobContext | None = None) -> str:
    """Make the brief available for delivery: SMTP if configured, otherwise an .eml in the outbox."""
    b = db.one("SELECT * FROM briefs WHERE id=?", (brief_id,))
    s = db.one("SELECT * FROM settings WHERE company_id=?", (company_id,)) or {}
    msg = EmailMessage()
    msg["Subject"] = b["title"]
    msg["From"] = config.SMTP_FROM or "signal-stack@localhost"
    msg["To"] = s.get("email_to") or "founder@example.invalid"
    msg.set_content(b["markdown"])
    msg.add_alternative(brief.email_html(b), subtype="html")
    if config.SMTP_HOST and s.get("email_to"):
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            smtp.starttls()
            if config.SMTP_USER:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.send_message(msg)
        how = f"emailed to {s['email_to']}"
    else:
        path = config.OUTBOX_DIR / f"brief_{brief_id}_{b['created_at'][:10]}.eml"
        path.write_bytes(bytes(msg))
        how = f"saved to outbox {path.name} (SMTP not configured)"
    db.update("briefs", brief_id, {"delivered_at": db.now(), "delivery": how})
    if ctx:
        ctx.step(f"Delivery: {how}")
    return how
