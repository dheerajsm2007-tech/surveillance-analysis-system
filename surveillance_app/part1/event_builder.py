import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from shared.models import TrackEvent


def build_events(all_frame_tracks: list) -> list[TrackEvent]:
    aggregated: dict[int, dict] = {}

    for frame_tracks in all_frame_tracks:
        for track in frame_tracks:
            tid = track.track_id
            if tid not in aggregated:
                aggregated[tid] = {
                    "track_id": tid,
                    "label": track.label,
                    "confidence": track.confidence,
                    "first_seen": track.first_seen,
                    "last_seen": track.last_seen,
                    "frame_count": track.frame_count,
                    "zones": list(track.zones),
                }
            else:
                agg = aggregated[tid]
                agg["last_seen"] = max(agg["last_seen"], track.last_seen)
                agg["first_seen"] = min(agg["first_seen"], track.first_seen)
                agg["frame_count"] = max(agg["frame_count"], track.frame_count)
                for z in track.zones:
                    if z not in agg["zones"]:
                        agg["zones"].append(z)
                agg["confidence"] = max(agg["confidence"], track.confidence)

    events = []
    for agg in aggregated.values():
        events.append(TrackEvent(
            track_id=agg["track_id"],
            label=agg["label"],
            confidence=agg["confidence"],
            first_seen=agg["first_seen"],
            last_seen=agg["last_seen"],
            dwell_seconds=agg["last_seen"] - agg["first_seen"],
            zones=agg["zones"],
            frame_count=agg["frame_count"],
        ))

    events.sort(key=lambda e: e.first_seen)
    return events


def detect_anomalies(events: list[TrackEvent]) -> list[str]:
    flags = []
    for ev in events:
        if ev.label == "person" and ev.dwell_seconds >= 240 and "exit-door" in ev.zones:
            flags.append(
                f"Person #{ev.track_id} loitered near exit-door for "
                f"{ev.dwell_seconds:.0f}s (entered at {ev.first_seen:.1f}s)"
            )
    return flags
