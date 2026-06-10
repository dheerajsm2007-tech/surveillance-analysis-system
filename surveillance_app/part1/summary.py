import sys
from datetime import datetime, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from shared.models import TrackEvent, VideoSummary


def build_summary(
    video_filename: str,
    duration_seconds: float,
    events: list[TrackEvent],
    anomaly_flags: list[str],
) -> VideoSummary:
    person_events = [e for e in events if e.label == "person"]
    total_unique = len(person_events)

    occupancy_by_ts: dict[float, int] = {}
    for ev in person_events:
        ts = round(ev.first_seen, 1)
        occupancy_by_ts[ts] = occupancy_by_ts.get(ts, 0) + 1

    peak_occ = 0
    peak_ts = 0.0
    for ts, cnt in occupancy_by_ts.items():
        if cnt > peak_occ:
            peak_occ = cnt
            peak_ts = ts

    return VideoSummary(
        video_filename=video_filename,
        duration_seconds=duration_seconds,
        total_unique_persons=total_unique,
        peak_occupancy=peak_occ,
        peak_occupancy_timestamp=peak_ts,
        events=events,
        anomaly_flags=anomaly_flags,
        processed_at=datetime.now(timezone.utc),
    )


def render_summary_text(summary: VideoSummary) -> str:
    SEP = "=" * 60
    SUB = "-" * 60

    person_events = [e for e in summary.events if e.label == "person"]
    dwells = [e.dwell_seconds for e in person_events]
    avg_dwell = sum(dwells) / len(dwells) if dwells else 0.0
    max_dwell = max(dwells) if dwells else 0.0

    dur_m = int(summary.duration_seconds // 60)
    dur_s = int(summary.duration_seconds % 60)

    lines = [
        SEP,
        "SURVEILLANCE ANALYSIS REPORT",
        SEP,
        f"  File       : {summary.video_filename}",
        f"  Duration   : {dur_m}m {dur_s:02d}s  ({summary.duration_seconds:.1f}s)",
        f"  Processed  : {summary.processed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        SUB,
        "AGGREGATE STATISTICS",
        SUB,
        f"  Individuals Detected : {summary.total_unique_persons}",
        f"  Peak Occupancy       : {summary.peak_occupancy} person(s) "
        f"at t={summary.peak_occupancy_timestamp:.1f}s",
        f"  Average Dwell Time   : {avg_dwell:.1f}s",
        f"  Longest Dwell Time   : {max_dwell:.1f}s",
        SUB,
        "INDIVIDUAL TRACKS",
        SUB,
    ]

    for ev in person_events:
        zones_str = ", ".join(ev.zones) if ev.zones else "unknown"
        lines.append(
            f"  Person #{ev.track_id:>3}  |  "
            f"entered {ev.first_seen:>6.1f}s  |  "
            f"exited {ev.last_seen:>6.1f}s  |  "
            f"dwell {ev.dwell_seconds:>6.1f}s  |  "
            f"zones: {zones_str}"
        )

    if not person_events:
        lines.append("  (no persons detected)")

    if summary.anomaly_flags:
        lines += [
            SUB,
            "ANOMALY FLAGS",
            SUB,
        ]
        for flag in summary.anomaly_flags:
            lines.append(f"  [!] {flag}")

    lines.append(SEP)
    return "\n".join(lines)
