from dataclasses import dataclass, field

MAX_LOST = 10
IOU_THRESH = 0.25


@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    bbox: list          # normalised [x1, y1, x2, y2]
    first_seen: float
    last_seen: float
    frame_count: int
    lost_frames: int
    zones: list = field(default_factory=list)


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _infer_zone(bbox: list) -> str:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    if cy > 0.75:
        return "exit-door"
    if cx < 0.33:
        return "entrance"
    if cx > 0.66:
        return "corridor"
    return "main-floor"


class SimpleTracker:
    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list, timestamp: float) -> list[Track]:
        matched_det = set()
        matched_trk = set()

        for ti, trk in enumerate(self._tracks):
            best_iou = IOU_THRESH
            best_di = -1
            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                if det.label != trk.label:
                    continue
                score = _iou(trk.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_di = di

            if best_di >= 0:
                det = detections[best_di]
                zone = _infer_zone(det.bbox)
                trk.bbox = det.bbox
                trk.confidence = det.confidence
                trk.last_seen = timestamp
                trk.frame_count += 1
                trk.lost_frames = 0
                if zone not in trk.zones:
                    trk.zones.append(zone)
                matched_det.add(best_di)
                matched_trk.add(ti)
            else:
                trk.lost_frames += 1

        for di, det in enumerate(detections):
            if di not in matched_det:
                zone = _infer_zone(det.bbox)
                self._tracks.append(Track(
                    track_id=self._next_id,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frame_count=1,
                    lost_frames=0,
                    zones=[zone],
                ))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.lost_frames <= MAX_LOST]
        return list(self._tracks)
