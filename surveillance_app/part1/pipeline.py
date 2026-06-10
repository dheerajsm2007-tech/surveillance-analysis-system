import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eagle" / "Embodied"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from part1.sampler import extract_frames, get_video_duration
from part1.detector import Detector
from part1.tracker import SimpleTracker
from part1.event_builder import build_events, detect_anomalies
from part1.summary import build_summary, render_summary_text
from shared.db import save_events, save_summary


def run_pipeline(video_path: str, save_to_db: bool = True, detector: Detector = None) -> dict:
    filename = Path(video_path).name

    print(f"[Pipeline] Getting duration for {filename}")
    duration = get_video_duration(video_path)

    print(f"[Pipeline] Extracting frames (duration={duration:.1f}s)")
    frames = extract_frames(video_path, target_fps=1.0)
    total = len(frames)
    print(f"[Pipeline] {total} frames sampled")

    if detector is None:
        print("[Pipeline] Loading detector...")
        detector = Detector()
        detector.load()

    tracker = SimpleTracker()
    all_frame_tracks = []

    for i, (timestamp, frame_bgr) in enumerate(frames):
        detections = detector.detect(frame_bgr)
        active_tracks = tracker.update(detections, timestamp)
        all_frame_tracks.append(list(active_tracks))

        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"[Pipeline] {i + 1}/{total} frames done")

    print("[Pipeline] Building events...")
    events = build_events(all_frame_tracks)
    anomaly_flags = detect_anomalies(events)

    print("[Pipeline] Building summary...")
    summary = build_summary(filename, duration, events, anomaly_flags)
    summary_text = render_summary_text(summary)

    if save_to_db:
        try:
            save_events(filename, events, summary.processed_at)
            save_summary(summary_text, summary, summary.processed_at)
            print("[Pipeline] Saved to database.")
        except Exception as exc:
            print(f"[Pipeline] DB save skipped: {exc}")

    return {
        "summary_text": summary_text,
        "total_persons": summary.total_unique_persons,
        "duration": duration,
        "peak_occupancy": summary.peak_occupancy,
        "anomaly_count": len(anomaly_flags),
        "anomaly_flags": anomaly_flags,
        "events": [e.model_dump() for e in events],
        "filename": filename,
    }
