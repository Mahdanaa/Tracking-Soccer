import cv2
import numpy as np
import supervision as sv

from sports.common.ball import BallAnnotator, BallTracker
from src.config import CONFIDENCE, IMAGE_SIZE, TILE_NMS, TILE_OVERLAP, TRAIL_LENGTH
from src.models import ModelStore


_Y, _X = np.ogrid[-1:1:17j, -1:1:17j]
_RADIUS_SQUARED = _X**2 + _Y**2
_CENTER = _RADIUS_SQUARED < 0.32**2
_RING = (_RADIUS_SQUARED > 0.61**2) & (_RADIUS_SQUARED < 0.90**2)


def candidate_contrast(gray: np.ndarray, detections: sv.Detections) -> np.ndarray:
    # An isolated bright center adds evidence for a ball, independent of the model score.
    contrasts = []
    for x1, y1, x2, y2 in detections.xyxy:
        size = (max(4, round((x2 - x1) * 1.8)), max(4, round((y2 - y1) * 1.8)))
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        patch = cv2.getRectSubPix(gray, size, center)
        patch = cv2.resize(patch, (17, 17), interpolation=cv2.INTER_LINEAR)
        contrasts.append(np.percentile(patch[_CENTER], 60) - np.percentile(patch[_RING], 65))
    return np.clip(np.asarray(contrasts) / 140, 0, 1)


class BallPipeline:
    def __init__(self, models: ModelStore, fps: float):
        self.models = models
        self.slicer = sv.InferenceSlicer(
            callback=self.detect_tile, slice_wh=IMAGE_SIZE, overlap_wh=TILE_OVERLAP,
            overlap_filter=sv.OverlapFilter.NONE, thread_workers=1,
        )
        self.tracker = BallTracker(fps=fps, acquisition_confidence=CONFIDENCE)
        self.trail = BallAnnotator(radius=7, buffer_size=TRAIL_LENGTH)
        self.previous_gray = None
        self.window = cv2.createHanningWindow((320, 180), cv2.CV_32F)

    def camera_shift(self, gray: np.ndarray) -> tuple[float, float]:
        height, width = gray.shape
        thumbnail = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        previous, self.previous_gray = self.previous_gray, thumbnail
        if previous is None:
            return 0.0, 0.0
        (dx, dy), response = cv2.phaseCorrelate(previous, thumbnail, self.window)
        if response < 0.20:
            if cv2.absdiff(previous, thumbnail).mean() > 22:
                self.tracker.reset()
                self.trail.buffer.clear()
            return 0.0, 0.0
        return dx * width / 320, dy * height / 180

    def detect_tile(self, frame: np.ndarray) -> sv.Detections:
        return sv.Detections.from_ultralytics(self.models.predict("ball", frame))

    def detect(self, frame: np.ndarray) -> sv.Detections:
        candidates = self.detect_tile(frame) if max(frame.shape[:2]) <= IMAGE_SIZE else self.slicer(frame)
        candidates = candidates.with_nms(threshold=TILE_NMS)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.tracker.update(candidates, self.camera_shift(gray),
                                   scale=1920 / max(gray.shape), contrast=candidate_contrast(gray, candidates))

    def draw(self, scene: np.ndarray, ball: sv.Detections) -> None:
        self.trail.annotate(scene, ball)
        for xy in ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int):
            cv2.circle(scene, tuple(xy), 10, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.circle(scene, tuple(xy), 10, (255, 255, 255), 2, cv2.LINE_AA)
