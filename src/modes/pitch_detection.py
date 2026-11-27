import numpy as np

from src.core.annotations import draw_landmarks
from src.core.pitch import detect_pitch


def run_pitch_detection(frames, run):
    for frame in frames:
        points = detect_pitch(frame, run.models)
        if not np.isfinite(points).any():
            run.warnings["No visible pitch landmarks"] += 1
        yield draw_landmarks(frame.copy(), points)
