"""Unit tests for the deterministic parts of Signal Stack (no network, no LLM)."""
import json
import os
import tempfile
from pathlib import Path

os.environ["SIGNALSTACK_DB"] = str(Path(tempfile.mkdtemp()) / "t.db")
os.environ["LLM_PROVIDER"] = "none"
os.environ["ENABLE_SCHEDULER"] = "0"

from signalstack import analysis, brief, db, internal, llm, textutil  # noqa: E402
from signalstack.adapters import _unwrap_bing, parse_feed  # noqa: E402

db.init_db()


def test_extract_json_handles_fences_and_prose():
    assert llm.extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert llm.extract_json('noise {"b": "x}y"} trailing') == {"b": "x}y"}


def test_repair_truncated_json_keeps_complete_prefix():
    cut = '{"summary": "ok", "offerings": ["a", "b"], "facts": [{"statement": "x"}, {"statement": "y'
    data = llm.repair_truncated_json(cut)
    assert data["summary"] == "ok" and data["offerings"] == ["a", "b"]
    assert data["facts"] == [{"statement": "x"}]


def test_canonical_url_strips_tracking_and_www():
    a = textutil.canonical_url("https://www.Example.com/news/x/?utm_source=t&id=3#frag")
    b = textutil.canonical_url("http://example.com/news/x?id=3")
    assert a == b == "https://example.com/news/x?id=3"


def test_parse_date_formats():
    assert textutil.parse_date("Tue, 24 Nov 2026 10:00:00 GMT").startswith("2026-11-24T10:00")
    assert textutil.parse_date("20261124T100000Z").startswith("2026-11-24T10:00")
    assert textutil.parse_date("garbage") is None


def test_bing_redirect_unwrapped():
    link = "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=x&url=https%3a%2f%2fexample.com%2fa&c=1"
    assert _unwrap_bing(link) == "https://example.com/a"


def test_parse_feed_rss():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
    <item><title>Canada caps study permits</title><link>https://ex.gov/a</link><pubDate>Mon, 23 Nov 2026 08:00:00 GMT</pubDate>
    <description>&lt;p&gt;New cap&lt;/p&gt;</description></item></channel></rss>"""
    recs = parse_feed(rss)
    assert recs[0]["title"] == "Canada caps study permits" and recs[0]["snippet"] == "New cap"
    assert recs[0]["published_at"].startswith("2026-11-23")


def test_topic_classification():
    assert analysis.classify_topic("Government announces new student visa rules and a cap") == "regulation"
    assert analysis.classify_topic("Rival raises Series B funding led by investors") == "funding"
    assert analysis.classify_topic("Company launches new app feature") == "product_launch"


def test_verification_and_credibility_are_independent_of_ai():
    assert analysis.verification_for("webpage", "https://rival.com/pricing", {}, 0.9) == "primary_source"
    assert analysis.verification_for("appstore_reviews", "https://apps.apple.com/x", {}, 0.6) == "user_generated"
    assert analysis.verification_for("bing_news", "https://unknown.blog/x", {"corroborations": 1}, 0.6) == "corroborated"
    assert analysis.publisher_credibility(0.7, "Reuters", "https://reuters.com/x", "bing_news") >= 0.85
    assert analysis.publisher_credibility(0.7, "random blog", "https://foo.xyz/x", "bing_news") == 0.6
    assert analysis.publisher_credibility(0.7, "", "https://www.gov.uk/x", "rss") == 0.95


def test_internal_classifier_and_csv_parse():
    assert internal.classify("We lost the deal, they went with a cheaper competitor") == "lost_deal"
    assert internal.classify("Too expensive for our budget this year") == "objection"
    assert internal.classify("Would love a mobile app with document tracking") == "product_request"
    kind, recs = internal.parse_upload("crm.csv", b"date,stage,notes\n2026-11-01,lost,Went with a cheaper competitor\n"
                                                  b"2026-11-02,won,Loved the counsellor follow up\n")
    assert kind == "csv" and len(recs) == 2 and recs[0]["meta"]["stage"] == "lost"


def test_brief_validation_drops_uncited_claims():
    data = {"executive_summary": [{"headline": "ok", "signal_ids": ["S1"]}, {"headline": "made up", "signal_ids": ["S999"]},
                                  {"headline": "no ids"}],
            "opportunities": [{"title": "x", "signal_ids": ["S1", "S2"]}], "risks": [], "actions": [], "watch": []}
    out = brief._validate(data, {"S1", "S2"})
    assert [x["headline"] for x in out["executive_summary"]] == ["ok"]
    assert out["opportunities"][0]["signal_ids"] == ["S1", "S2"]


def test_key_phrases_and_similarity():
    texts = ["study abroad counselling for Indian students", "study abroad counselling and visa support"]
    assert "study abroad" in " ".join(textutil.key_phrases(texts))
    sims = textutil.sim_to_reference("study abroad visa counselling", ["visa counselling for study abroad", "cricket score today"])
    assert sims[0] > sims[1]


# --------------------------------------------------------------------------- LLM failover, grounding, scheduling
class _Resp:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = text or json.dumps(payload or {})

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_llm_fails_over_to_next_model_when_daily_quota_is_spent(monkeypatch):
    p1 = llm.Provider("groq", "https://api.groq.com/openai/v1", "model-a", "k", 0)
    p2 = llm.Provider("groq", "https://api.groq.com/openai/v1", "model-b", "k", 0)
    monkeypatch.setattr(llm, "providers", lambda purpose="": [p1, p2])
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "model-a":
            return _Resp(429, text="Rate limit reached on tokens per day (TPD): Limit 100000. Please try again in 1h2m3s.")
        return _Resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    text, label = llm.chat("sys", "user-failover-test", purpose="t", use_cache=False)
    assert text == '{"ok": true}' and label == "groq:model-b"
    assert calls == ["model-a", "model-b"]
    assert llm._disabled_until["groq:model-a"] > 0  # quota-exhausted model is parked, not retried every call


def test_parse_wait_reads_groq_message():
    assert llm._parse_wait("Please try again in 1h2m3s.", None) == 3723
    assert llm._parse_wait("try again in 7.5s", None) == 7.5
    assert llm._parse_wait("", "12") == 12


def test_grounding_check_removes_unsupported_and_corrects_partial(monkeypatch):
    syn = {"executive_summary": [{"headline": "A", "detail": "x", "signal_ids": ["S1"]},
                                 {"headline": "B", "detail": "invented", "signal_ids": ["S2"]}],
           "opportunities": [{"title": "O", "evidence": "overstated", "impact": "i", "signal_ids": ["S1"]}],
           "risks": [], "actions": [], "watch": []}
    by_sid = {"S1": {"title": "t1", "snippet": "s1"}, "S2": {"title": "t2", "snippet": "s2"}}
    answer = {"checks": [{"id": "EXE1", "supported": "yes"}, {"id": "EXE2", "supported": "no", "note": "not in S2"},
                         {"id": "OPP1", "supported": "partial", "fix": "only what S1 says"}]}
    monkeypatch.setattr(brief.llm, "chat_json", lambda *a, **k: (answer, "groq:test"))
    g = brief._grounding_check(syn, by_sid)
    assert g["checked"] == 3 and g["removed"] == 1 and g["partial"] == 1
    assert [x["headline"] for x in syn["executive_summary"]] == ["A"]
    assert syn["opportunities"][0]["evidence"] == "only what S1 says"


def test_missed_scheduled_run_is_logged():
    from signalstack import scheduler
    cid = db.insert("companies", {"name": "MissCo", "website": "https://miss.example", "domain": "miss.example",
                                  "status": "stack_ready", "created_at": "2020-01-01T00:00:00+00:00",
                                  "updated_at": "2020-01-01T00:00:00+00:00"})
    db.insert("settings", {"company_id": cid, "updated_at": db.now()})
    missed = scheduler.check_missed_runs(catch_up=False)
    jobs = {m["job"] for m in missed if m["company_id"] == cid}
    assert jobs == {"weekly_brief", "monthly_audit"}
    rows = db.q("SELECT status FROM job_runs WHERE company_id=? AND status='missed'", (cid,))
    assert len(rows) == 2
    assert all(m["company_id"] != cid for m in scheduler.check_missed_runs(catch_up=False))  # logged once, not every start
