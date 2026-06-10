import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/surveillance",
)

db_available = False
engine = None
SessionLocal = None
Base = None

try:
    from sqlalchemy import (
        Column, Float, Integer, String, Text, DateTime, JSON, create_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, sessionmaker

    class _Base(DeclarativeBase):
        pass

    Base = _Base

    class EventRecord(_Base):
        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        video_filename = Column(Text, nullable=False)
        track_id = Column(Integer, nullable=False)
        label = Column(Text, nullable=False)
        confidence = Column(Float, nullable=False)
        first_seen = Column(Float, nullable=False)
        last_seen = Column(Float, nullable=False)
        dwell_seconds = Column(Float, nullable=False)
        zones = Column(JSON, nullable=False, default=list)
        frame_count = Column(Integer, nullable=False)
        processed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    class SummaryRecord(_Base):
        __tablename__ = "summaries"
        id = Column(Integer, primary_key=True)
        video_filename = Column(Text, nullable=False)
        duration_seconds = Column(Float, nullable=False)
        total_unique_persons = Column(Integer, nullable=False)
        peak_occupancy = Column(Integer, nullable=False)
        peak_occupancy_timestamp = Column(Float, nullable=False)
        anomaly_flags = Column(JSON, nullable=False, default=list)
        summary_text = Column(Text, nullable=False)
        processed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)

    with engine.connect() as conn:
        pass

    db_available = True
    print(f"[DB] Connected to {DATABASE_URL.split('@')[-1]}")

except Exception as exc:
    print(f"[DB] Warning: database unavailable — {exc}. Running without persistence.")


def init_db():
    if not db_available:
        print("[DB] Skipping init_db — no database connection.")
        return
    Base.metadata.create_all(bind=engine)


def save_events(video_filename: str, events, processed_at: datetime):
    if not db_available:
        return
    try:
        session = SessionLocal()
        for ev in events:
            session.add(EventRecord(
                video_filename=video_filename,
                track_id=ev.track_id,
                label=ev.label,
                confidence=ev.confidence,
                first_seen=ev.first_seen,
                last_seen=ev.last_seen,
                dwell_seconds=ev.dwell_seconds,
                zones=ev.zones,
                frame_count=ev.frame_count,
                processed_at=processed_at,
            ))
        session.commit()
        session.close()
    except Exception as exc:
        print(f"[DB] save_events error: {exc}")


def save_summary(summary_text: str, summary, processed_at: datetime):
    if not db_available:
        return
    try:
        session = SessionLocal()
        session.add(SummaryRecord(
            video_filename=summary.video_filename,
            duration_seconds=summary.duration_seconds,
            total_unique_persons=summary.total_unique_persons,
            peak_occupancy=summary.peak_occupancy,
            peak_occupancy_timestamp=summary.peak_occupancy_timestamp,
            anomaly_flags=summary.anomaly_flags,
            summary_text=summary_text,
            processed_at=processed_at,
        ))
        session.commit()
        session.close()
    except Exception as exc:
        print(f"[DB] save_summary error: {exc}")
