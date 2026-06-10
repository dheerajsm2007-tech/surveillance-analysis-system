import cv2
import numpy as np


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps


def extract_frames(video_path: str, target_fps: float = 1.0) -> list:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sample_interval = src_fps / target_fps  # src frames between samples
    min_gap_frames = int(0.5 * src_fps)    # 0.5s minimum gap

    frames = []
    prev_gray = None
    last_sampled = -min_gap_frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / src_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gap_ok = (frame_idx - last_sampled) >= min_gap_frames
        due_to_fps = (frame_idx % max(1, int(sample_interval))) == 0

        scene_change = False
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
            scene_change = diff > 25.0

        if gap_ok and (due_to_fps or scene_change):
            frames.append((timestamp, frame.copy()))
            last_sampled = frame_idx

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return frames
