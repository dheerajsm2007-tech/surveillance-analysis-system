import os
import re
import sys
from dataclasses import dataclass, field

import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "eagle" / "Embodied"))

# Use local path if set (avoids HF network call); falls back to HF hub (loads from cache)
MODEL_ID = os.environ.get(
    "LOCATEANYTHING_MODEL_PATH",
    r"C:\Users\dheer\OneDrive\Documents\GitHub\LocateAnything-3B",
)
CATEGORIES = ["person", "vehicle", "bag", "backpack", "suitcase"]
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list  # normalised [x1, y1, x2, y2] in [0, 1]


def _parse_answer(answer: str, categories: list, img_w: int, img_h: int) -> list:
    """Parse raw model answer text into Detection objects.

    The model outputs text with <box><x1><y1><x2><y2></box> tags where
    coordinates are integers in [0, 1000]. Category labels appear in the
    text before each box. Normalised bbox is returned in [0, 1].
    """
    box_re = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
    cat_re = re.compile("|".join(re.escape(c) for c in categories), re.IGNORECASE)

    segments = re.split(r"(<box><\d+><\d+><\d+><\d+></box>)", answer)

    detections = []
    current_label = "person"
    for seg in segments:
        m = box_re.fullmatch(seg)
        if m:
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            detections.append(Detection(
                label=current_label,
                confidence=0.85,
                bbox=[x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0],
            ))
        else:
            found = cat_re.findall(seg)
            if found:
                current_label = found[-1].lower()

    return detections


class Detector:
    def __init__(self):
        self.worker = None

    def load(self):
        from locateanything_worker import LocateAnythingWorker
        self.worker = LocateAnythingWorker(MODEL_ID, device=device, dtype=torch_dtype)

    def detect(self, frame_bgr) -> list:
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]

        result = self.worker.detect(pil_img, CATEGORIES)
        answer = result.get("answer", "")

        detections = _parse_answer(answer, CATEGORIES, w, h)
        return [d for d in detections if d.confidence >= 0.4]
