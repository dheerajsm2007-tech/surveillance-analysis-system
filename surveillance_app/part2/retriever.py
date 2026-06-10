import json
import os
import re

import numpy as np
import psycopg2
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/surveillance",
)


def _parse_time_bounds(question: str):
    """
    Extracts time bounds from natural language.
    Returns (lower_seconds, upper_seconds) or (None, None) if no time found.
    Handles patterns like: after 10pm, before 2pm, between 2 and 3pm.
    """
    question_lower = question.lower()

    def to_seconds(hour, meridiem):
        h = int(hour)
        if meridiem == "pm" and h != 12:
            h += 12
        if meridiem == "am" and h == 12:
            h = 0
        return h * 3600

    # between X and Y
    m = re.search(r"between\s+(\d+)\s*(?:am|pm)?\s+and\s+(\d+)\s*(am|pm)?", question_lower)
    if m:
        mer = m.group(3) or "pm"
        return to_seconds(m.group(1), mer), to_seconds(m.group(2), mer)

    # after X
    m = re.search(r"after\s+(\d+)\s*(am|pm)?", question_lower)
    if m:
        mer = m.group(2) or "pm"
        return to_seconds(m.group(1), mer), None

    # before X
    m = re.search(r"before\s+(\d+)\s*(am|pm)?", question_lower)
    if m:
        mer = m.group(2) or "pm"
        return None, to_seconds(m.group(1), mer)

    return None, None


def _cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def retrieve(question: str, video_filename: str, top_k: int = 5) -> list[Document]:
    """
    Two-stage retrieval:
    1. SQL time filter (if question contains a time reference)
    2. Python-side cosine similarity on embeddings stored as JSON text
    Returns list of LangChain Document objects ranked by similarity.
    """
    lower_ts, upper_ts = _parse_time_bounds(question)

    # Stage 1 — SQL query with optional time filter
    conditions = ["video_filename = %s", "embedding IS NOT NULL"]
    params: list = [video_filename]

    if lower_ts is not None:
        conditions.append("first_seen >= %s")
        params.append(lower_ts)
    if upper_ts is not None:
        conditions.append("first_seen <= %s")
        params.append(upper_ts)

    where = " AND ".join(conditions)
    query = f"""
        SELECT track_id, label, confidence, first_seen, last_seen,
               dwell_seconds, zones, video_filename, embedding
        FROM events
        WHERE {where}
    """

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return []

    # Stage 2 — embed the question and rank rows by cosine similarity
    model = OllamaEmbeddings(model="nomic-embed-text")
    q_vec = model.embed_query(question)

    scored = []
    for row in rows:
        track_id, label, confidence, first_seen, last_seen, \
        dwell_seconds, zones, vid_filename, embedding_json = row

        try:
            ev_vec = json.loads(embedding_json)
        except (TypeError, json.JSONDecodeError):
            continue

        similarity = _cosine_similarity(q_vec, ev_vec)
        scored.append((similarity, track_id, label, confidence, first_seen,
                       last_seen, dwell_seconds, zones, vid_filename))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    docs = []
    for sim, track_id, label, confidence, first_seen, last_seen, \
            dwell_seconds, zones, vid_filename in top:

        zones_list = zones if isinstance(zones, list) else json.loads(zones or "[]")

        def fmt_ts(s):
            return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"

        def fmt_dur(s):
            return f"{int(s // 60)}m {int(s % 60):02d}s"

        text = (
            f"{label.capitalize()} #{track_id:03d}: "
            f"entered {fmt_ts(first_seen)}, exited {fmt_ts(last_seen)}, "
            f"dwell {fmt_dur(dwell_seconds)}, "
            f"zones: {', '.join(zones_list)}, "
            f"confidence {confidence:.2f}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "track_id":   track_id,
                "label":      label,
                "similarity": round(sim, 3),
            },
        ))
    return docs
