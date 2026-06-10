import os

import psycopg2
from dotenv import load_dotenv
from flask import Blueprint, jsonify, request

from part2.chain import ask
from part2.embedder import embed_events_for_video

load_dotenv()

qa = Blueprint("qa", __name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/surveillance",
)


@qa.route("/api/videos", methods=["GET"])
def list_videos():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            SELECT video_filename, processed_at, total_unique_persons
            FROM summaries
            ORDER BY processed_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([
            {
                "video_filename":       r[0],
                "processed_at":         r[1].isoformat() if r[1] else None,
                "total_unique_persons": r[2],
            }
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@qa.route("/api/embed", methods=["POST"])
def embed_video():
    data     = request.get_json()
    filename = data.get("video_filename")
    if not filename:
        return jsonify({"error": "video_filename required"}), 400
    try:
        n = embed_events_for_video(filename)
        return jsonify({"embedded": n, "message": f"Embedded {n} events for {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@qa.route("/api/ask", methods=["POST"])
def ask_question():
    data     = request.get_json()
    question = data.get("question", "").strip()
    filename = data.get("video_filename", "").strip()
    if not question or not filename:
        return jsonify({"error": "Both question and video_filename are required"}), 400
    try:
        result = ask(question, filename)
        return jsonify(result)
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ("connection", "refused", "ollama", "connect error")):
            return jsonify({"error": "Ollama is not running. Start it with: ollama serve"}), 503
        return jsonify({"error": msg}), 500
