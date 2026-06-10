from datetime import datetime
from pydantic import BaseModel


class TrackEvent(BaseModel):
    track_id: int
    label: str
    confidence: float
    first_seen: float
    last_seen: float
    dwell_seconds: float
    zones: list[str]
    frame_count: int


class VideoSummary(BaseModel):
    video_filename: str
    duration_seconds: float
    total_unique_persons: int
    peak_occupancy: int
    peak_occupancy_timestamp: float
    events: list[TrackEvent]
    anomaly_flags: list[str]
    processed_at: datetime
