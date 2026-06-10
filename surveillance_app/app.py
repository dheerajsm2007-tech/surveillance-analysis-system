import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eagle" / "Embodied"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from part1.detector import Detector
from part1.pipeline import run_pipeline
from shared.db import SessionLocal, db_available

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "surveillance_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

from part2.qa_routes import qa  # noqa: E402
app.register_blueprint(qa)


@app.route("/qa")
def qa_page():
    return render_template("qa.html")

# Pre-load model once at startup
_detector = Detector()
print("[App] Loading LocateAnything model at startup...")
try:
    _detector.load()
    print("[App] Model ready.")
except Exception as exc:
    print(f"[App] WARNING: model failed to load — {exc}")
    print("[App] Pipeline will attempt reload on first request.")

JOBS: dict[str, dict] = {}


def _run_job(job_id: str, video_path: str):
    JOBS[job_id]["status"] = "processing"
    try:
        if _detector.worker is None:
            print("[App] Detector not loaded; retrying load...")
            _detector.load()
        result = run_pipeline(video_path, save_to_db=True, detector=_detector)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        print(f"[App] Job {job_id} error: {exc}")
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    f.save(str(save_path))

    JOBS[job_id] = {"status": "queued", "result": None, "error": None}
    thread = threading.Thread(target=_run_job, args=(job_id, str(save_path)), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route("/history")
def history():
    if not db_available:
        return jsonify({"warning": "Database not connected — history unavailable", "items": []})
    try:
        from shared.db import SummaryRecord
        session = SessionLocal()
        rows = (
            session.query(SummaryRecord)
            .order_by(SummaryRecord.processed_at.desc())
            .limit(10)
            .all()
        )
        session.close()
        items = [
            {
                "filename": r.video_filename,
                "duration": r.duration_seconds,
                "persons": r.total_unique_persons,
                "anomalies": len(r.anomaly_flags or []),
                "processed_at": r.processed_at.strftime("%Y-%m-%d %H:%M UTC") if r.processed_at else "",
            }
            for r in rows
        ]
        return jsonify({"items": items})
    except Exception as exc:
        return jsonify({"warning": str(exc), "items": []})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
