import json
import os

import psycopg2
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/surveillance",
)


def get_embeddings_model():
    return OllamaEmbeddings(model="nomic-embed-text")


def build_event_text(row: dict) -> str:
    """
    Convert a raw event DB row into a text chunk for embedding.
    Example output:
    Person #003 entered at 00:02:10, exited at 00:08:41, dwell 6m 31s.
    Zones visited: entrance, exit-door. Confidence: 0.94.
    Video: shop_floor_cam1.mp4
    """
    zones = row["zones"] if isinstance(row["zones"], list) else json.loads(row["zones"] or "[]")
    zones_str = ", ".join(zones) if zones else "unclassified"
    first = row["first_seen"]
    last  = row["last_seen"]
    dwell = row["dwell_seconds"]

    def fmt_ts(s):
        return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"

    def fmt_dur(s):
        return f"{int(s // 60)}m {int(s % 60):02d}s"

    return (
        f"{row['label'].capitalize()} #{row['track_id']:03d} "
        f"entered at {fmt_ts(first)}, exited at {fmt_ts(last)}, "
        f"dwell {fmt_dur(dwell)}. "
        f"Zones visited: {zones_str}. "
        f"Confidence: {row['confidence']:.2f}. "
        f"Video: {row['video_filename']}"
    )


def embed_events_for_video(video_filename: str) -> int:
    """
    Embeds all events for a given video that do not yet have an embedding.
    Returns number of rows embedded.
    Embeddings stored as JSON-encoded float list in the TEXT column.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    cur.execute(
        """
        SELECT id, track_id, label, confidence, first_seen, last_seen,
               dwell_seconds, zones, video_filename
        FROM events
        WHERE video_filename = %s AND embedding IS NULL
        """,
        (video_filename,),
    )
    rows = cur.fetchall()

    if not rows:
        cur.close()
        conn.close()
        return 0

    columns = [
        "id", "track_id", "label", "confidence", "first_seen",
        "last_seen", "dwell_seconds", "zones", "video_filename",
    ]
    records = [dict(zip(columns, row)) for row in rows]

    model   = get_embeddings_model()
    texts   = [build_event_text(r) for r in records]
    vectors = model.embed_documents(texts)

    for record, vector in zip(records, vectors):
        cur.execute(
            "UPDATE events SET embedding = %s WHERE id = %s",
            (json.dumps(vector), record["id"]),
        )

    conn.commit()
    cur.close()
    conn.close()
    return len(records)
