"""Internal intelligence: CSV / Excel / PDF / text uploads -> records -> themes.

Each row / paragraph / speaker turn becomes an item in the 'internal' category, classified
as objection, lost-deal reason, product request, complaint, praise, question or operational
issue, and clustered into recurring themes (TF-IDF + KMeans)."""
from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, db, llm
from .profile import profile_summary_for_prompt
from .textutil import STOP, clean_ws, parse_date, sha, truncate

LABELS = [
    ("lost_deal", r"\b(lost|went with|chose (another|competitor)|closed[- ]lost|no decision|churn\w*|cancel\w*|didn'?t (buy|convert)|dropped out)\b"),
    ("objection", r"\b(too expensive|expensive|price|budget|not sure|concern\w*|worried|hesitant|trust|think about it|"
                  r"cheaper|discount|objection)\b"),
    ("product_request", r"\b(wish|would like|would love|please add|feature request|need(s|ed)? (a|an|the|more)|missing|"
                        r"should (have|offer|add)|can you add|request\w*)\b"),
    ("complaint", r"\b(complain\w*|slow|delay\w*|bad|poor|terrible|rude|no response|not respond\w*|issue|problem|bug|"
                  r"broken|wrong|refund)\b"),
    ("praise", r"\b(great|excellent|helpful|love|amazing|thank\w*|recommend\w*|smooth|happy)\b"),
    ("question", r"\?\s*$|\b(how (do|can|long)|what is|is it possible|when will)\b"),
    ("operational", r"\b(process|follow[- ]?up|handover|staff\w*|capacity|backlog|sla|response time|documentation)\b"),
]
LABEL_RX = [(l, re.compile(rx, re.I)) for l, rx in LABELS]
TEXT_COL_HINTS = re.compile(r"note|comment|feedback|reason|transcript|message|summary|description|text|remark|objection|request", re.I)
DATE_COL_HINTS = re.compile(r"date|created|time|closed|updated", re.I)
LABEL_COL_HINTS = re.compile(r"stage|status|outcome|result|reason|type|category|segment|source|country|program|plan", re.I)


def classify(text: str) -> str:
    for label, rx in LABEL_RX:
        if rx.search(text):
            return label
    return "other"


def _records_from_table(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(how="all")
    str_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        return []
    avg_len = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
    text_cols = [c for c in str_cols if TEXT_COL_HINTS.search(str(c))] or [max(avg_len, key=avg_len.get)]
    date_col = next((c for c in df.columns if DATE_COL_HINTS.search(str(c))), None)
    label_cols = [c for c in df.columns if LABEL_COL_HINTS.search(str(c)) and c not in text_cols][:4]
    out = []
    for i, row in df.iterrows():
        text = " | ".join(clean_ws(str(row[c])) for c in text_cols if pd.notna(row[c]) and str(row[c]).strip())
        if len(text) < 3:
            continue
        meta = {str(c): clean_ws(str(row[c])) for c in label_cols if pd.notna(row[c])}
        out.append({"text": text, "meta": meta, "date": parse_date(str(row[date_col])) if date_col and pd.notna(row[date_col]) else None,
                    "ref": f"row {int(i) + 2}"})
    return out


def _records_from_text(text: str) -> list[dict]:
    # speaker turns ("Name: ...") or paragraphs
    chunks = re.split(r"\n\s*\n|\n(?=[A-Z][A-Za-z .'-]{1,30}:\s)", text)
    out = []
    for n, ch in enumerate(chunks, 1):
        ch = clean_ws(ch)
        if len(ch) < 25:
            continue
        for part in [ch[i:i + 900] for i in range(0, len(ch), 900)]:
            out.append({"text": part, "meta": {}, "date": None, "ref": f"para {n}"})
    return out


def parse_upload(filename: str, data: bytes) -> tuple[str, list[dict]]:
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return "csv", _records_from_table(pd.read_csv(io.BytesIO(data), encoding_errors="replace"))
    if ext in (".xlsx", ".xlsm", ".xls"):
        frames = pd.read_excel(io.BytesIO(data), sheet_name=None)
        recs = []
        for name, df in frames.items():
            for r in _records_from_table(df):
                r["ref"] = f"{name} {r['ref']}"
                recs.append(r)
        return "xlsx", recs
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((pg.extract_text() or "") for pg in reader.pages)
        return "pdf", _records_from_text(text)
    if ext in (".txt", ".md", ".vtt", ".srt"):
        return "txt", _records_from_text(data.decode("utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type {ext}. Use CSV, Excel, PDF or text.")


THEME_SYSTEM = """You summarise internal business data (CRM notes, call transcripts, support tickets) for a founder.
Only state patterns that are supported by the records; quote counts given to you; no invented numbers."""
THEME_PROMPT = """Company: {profile}

Record counts by label: {counts}
Clusters (top terms + example records):
{clusters}

Return JSON: {{"themes": [{{"theme": "...", "label": "objection|lost_deal|product_request|complaint|operational|praise",
"count": <records in cluster>, "insight": "what it means for the business", "example": "short quote"}}],
"emerging_risks": ["..."]}}"""


def ingest_upload(company_id: int, filename: str, data: bytes, label: str = "other", is_synthetic: bool = False) -> dict:
    kind, recs = parse_upload(filename, data)
    if not recs:
        raise ValueError("No usable text records found in the file.")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    path = config.UPLOAD_DIR / f"{company_id}_{sha(safe + db.now())[:8]}_{safe}"
    path.write_bytes(data)
    source = db.one("SELECT * FROM sources WHERE company_id=? AND type='upload' LIMIT 1", (company_id,))
    if not source:
        sid = db.insert("sources", {"company_id": company_id, "category": "internal", "type": "upload",
                                    "name": "Internal uploads inbox (CRM, calls, support, feedback)",
                                    "reason": "User-provided internal data", "status": "approved", "origin": "user",
                                    "credibility_score": 0.8, "created_at": db.now(), "approved_at": db.now()})
        source = db.one("SELECT * FROM sources WHERE id=?", (sid,))
    upload_id = db.insert("uploads", {"company_id": company_id, "source_id": source["id"], "filename": filename, "kind": kind,
                                      "label": label, "is_synthetic": int(is_synthetic), "records": len(recs),
                                      "uploaded_at": db.now()})
    run_id = db.insert("crawl_runs", {"company_id": company_id, "trigger": "upload", "started_at": db.now(),
                                      "finished_at": db.now(), "status": "success", "sources_checked": 1,
                                      "items_found": len(recs), "items_new": len(recs)})
    labels = Counter()
    prefix = "[SYNTHETIC TEST DATA] " if is_synthetic else ""
    for n, r in enumerate(recs):
        lab = classify(r["text"])
        labels[lab] += 1
        meta_s = ", ".join(f"{k}: {v}" for k, v in r["meta"].items())
        db.execute(
            "INSERT OR IGNORE INTO items (company_id, source_id, run_id, category, url, url_hash, title, snippet, publisher, "
            "published_at, discovered_at, content_hash, topic, analysis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (company_id, source["id"], run_id, "internal", f"upload://{upload_id}/{r['ref'].replace(' ', '-')}",
             sha(f"upload:{upload_id}:{n}"), prefix + f"{lab.replace('_', ' ').title()}: {truncate(r['text'], 140)}",
             r["text"] + (f" ({meta_s})" if meta_s else ""), f"{filename} ({r['ref']})", r["date"], db.now(),
             sha(r["text"]), lab, json.dumps({"internal_label": lab, "meta": r["meta"], "synthetic": is_synthetic})))
        db.execute("UPDATE sources SET items_total = items_total + 1, last_new_item_at=?, last_success=?, last_checked=? WHERE id=?",
                   (db.now(), db.now(), db.now(), source["id"]))
    summary = summarise_internal(company_id, upload_id, recs, labels)
    db.update("uploads", upload_id, {"summary": summary})
    return {"upload_id": upload_id, "records": len(recs), "labels": dict(labels), "summary": summary}


def summarise_internal(company_id: int, upload_id: int, recs: list[dict], labels: Counter) -> dict:
    texts = [r["text"] for r in recs]
    clusters = []
    if len(texts) >= 6:
        k = max(2, min(6, len(texts) // 5))
        vec = TfidfVectorizer(stop_words=list(STOP), ngram_range=(1, 2), min_df=1, max_features=5000)
        X = vec.fit_transform(texts)
        km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
        terms = vec.get_feature_names_out()
        for c in range(k):
            idx = [i for i, l in enumerate(km.labels_) if l == c]
            top = [terms[j] for j in km.cluster_centers_[c].argsort()[::-1][:6]]
            lab = Counter(classify(texts[i]) for i in idx).most_common(1)[0][0]
            clusters.append({"size": len(idx), "terms": top, "label": lab, "examples": [truncate(texts[i], 160) for i in idx[:3]]})
        clusters.sort(key=lambda c: -c["size"])
    data, used = llm.chat_json(THEME_SYSTEM, THEME_PROMPT.format(
        profile=profile_summary_for_prompt(company_id, None, 1200), counts=dict(labels),
        clusters="\n".join(f"- size {c['size']}, label {c['label']}, terms {c['terms']}, e.g. {c['examples']}" for c in clusters)
        or "\n".join("- " + truncate(t, 160) for t in texts[:20])), purpose="internal_themes")
    themes = data.get("themes", []) if isinstance(data, dict) else [
        {"theme": ", ".join(c["terms"][:3]), "label": c["label"], "count": c["size"],
         "insight": f"{c['size']} records cluster around {', '.join(c['terms'][:3])}", "example": c["examples"][0]} for c in clusters]
    risks = data.get("emerging_risks", []) if isinstance(data, dict) else []
    return {"labels": dict(labels), "clusters": clusters, "themes": themes, "emerging_risks": risks, "by": used}
