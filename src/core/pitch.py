import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from src.config import (COLORS, HOMOGRAPHY_FALLBACK_TOLERANCE_PX, HOMOGRAPHY_MODEL_TOLERANCE_PX,
                        HOMOGRAPHY_SMOOTHING, IMAGE_SIZE, KEYPOINT_CONFIDENCE, KEYPOINT_FALLBACK_CONFIDENCE)
from src.models import ModelStore

PITCH = SoccerPitchConfiguration()
VERTICES = np.asarray(PITCH.vertices, dtype=np.float32)


def _extract_keypoints(frame: np.ndarray, models: ModelStore) -> tuple[np.ndarray, np.ndarray] | None:
    result = models.predict("pitch", frame)
    keypoints = result.keypoints
    if keypoints is None or tuple(keypoints.xy.shape[1:]) != (32, 2):
        raise ValueError("Expected pitch keypoints with shape (N,32,2).")
    if not len(keypoints.xy):
        return None
    best = int(result.boxes.conf.argmax())
    xy = keypoints.xy[best].cpu().numpy()
    if keypoints.conf is None or keypoints.conf.shape != keypoints.xy.shape[:2]:
        raise ValueError("Expected confidence for every pitch keypoint.")
    confidence = keypoints.conf[best].cpu().numpy()
    return xy, confidence


def _filter_keypoints(extracted: tuple[np.ndarray, np.ndarray] | None, min_confidence: float) -> np.ndarray:
    points = np.full((32, 2), np.nan, dtype=np.float32)
    if extracted is None:
        return points
    xy, confidence = extracted
    valid = np.isfinite(xy).all(axis=1) & (xy > 0).all(axis=1)
    valid &= np.isfinite(confidence) & (confidence >= min_confidence)
    points[valid] = xy[valid]
    return points


def detect_pitch(frame: np.ndarray, models: ModelStore) -> np.ndarray:
    return _filter_keypoints(_extract_keypoints(frame, models), KEYPOINT_CONFIDENCE)


def build_transform(points: np.ndarray, frame_shape: tuple[int, ...],
                    tolerance_px: float = HOMOGRAPHY_MODEL_TOLERANCE_PX) -> ViewTransformer | None:
    if points.shape != VERTICES.shape:
        raise ValueError("Expected 32 ordered pitch coordinates.")
    valid = np.isfinite(points).all(axis=1)
    source, target = points[valid], VERTICES[valid]
    if len(source) < 4:
        return None
    if any(np.linalg.matrix_rank(p - p.mean(axis=0)) < 2 for p in (source, target)):
        return None
    try:
        # Convert the error allowance from letterboxed model pixels to source pixels.
        tolerance = tolerance_px * max(frame_shape[:2]) / IMAGE_SIZE
        _, mask = cv2.findHomography(target, source, cv2.RANSAC, tolerance)
        required = max(4, len(source) / 2)
        if mask is None or mask.sum() < required:
            # Local optimization can recover consensus during a pan using this frame alone.
            _, mask = cv2.findHomography(target, source, cv2.USAC_DEFAULT, tolerance)
        if mask is None or mask.sum() < required:
            return None
        keep = mask.ravel().astype(bool)
        transform = ViewTransformer(source[keep], target[keep])
    except (ValueError, cv2.error):
        return None
    matrix = transform.m
    return transform if np.isfinite(matrix).all() and np.linalg.matrix_rank(matrix) == 3 else None


def detect_pitch_transform(frame: np.ndarray, models: ModelStore) -> ViewTransformer | None:
    extracted = _extract_keypoints(frame, models)
    if extracted is None:
        return None
    strict = build_transform(_filter_keypoints(extracted, KEYPOINT_CONFIDENCE), frame.shape,
                             HOMOGRAPHY_MODEL_TOLERANCE_PX)
    if strict is not None:
        return strict
    return build_transform(_filter_keypoints(extracted, KEYPOINT_FALLBACK_CONFIDENCE), frame.shape,
                           HOMOGRAPHY_FALLBACK_TOLERANCE_PX)


def _normalize_homography(matrix: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        return None
    scale = matrix[2, 2]
    if abs(scale) < 1e-7:
        return None
    return (matrix / scale).astype(np.float64)


def _transform_from_matrix(matrix: np.ndarray) -> ViewTransformer:
    transform = object.__new__(ViewTransformer)
    transform.m = matrix
    return transform


class PitchStabilizer:
    def __init__(self, smoothing: float = HOMOGRAPHY_SMOOTHING):
        self.smoothing = smoothing
        self.matrix: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.window = cv2.createHanningWindow((320, 180), cv2.CV_32F)

    def _camera_shift(self, frame: np.ndarray) -> tuple[float, float, bool]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        previous, self.previous_gray = self.previous_gray, thumbnail
        if previous is None:
            return 0.0, 0.0, False
        (dx, dy), response = cv2.phaseCorrelate(previous, thumbnail, self.window)
        if response < 0.20:
            scene_cut = float(cv2.absdiff(previous, thumbnail).mean()) > 22
            return 0.0, 0.0, scene_cut
        return dx * width / 320, dy * height / 180, False

    def _probe_points(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        return np.array(
            [[width * 0.25, height * 0.5], [width * 0.75, height * 0.5],
             [width * 0.5, height * 0.35], [width * 0.5, height * 0.65]],
            dtype=np.float32,
        )

    def update(self, frame: np.ndarray, transform: ViewTransformer | None) -> ViewTransformer | None:
        dx, dy, scene_cut = self._camera_shift(frame)
        candidate = _normalize_homography(transform.m) if transform is not None else None
        if candidate is not None:
            if self.matrix is not None and not scene_cut:
                probes = self._probe_points(frame.shape)
                old_pts = map_points(_transform_from_matrix(self.matrix), probes)
                new_pts = map_points(_transform_from_matrix(candidate), probes)
                valid = np.isfinite(old_pts).all(axis=1) & np.isfinite(new_pts).all(axis=1)
                if valid.any() and float(np.median(np.linalg.norm(new_pts[valid] - old_pts[valid], axis=1))) < 1800:
                    blended = _normalize_homography(self.smoothing * candidate + (1 - self.smoothing) * self.matrix)
                    if blended is not None:
                        candidate = blended
            self.matrix = candidate
            return _transform_from_matrix(self.matrix)

        if self.matrix is None:
            return None

        # Keep the last valid calibration active and compensate for smooth camera pan.
        if dx or dy:
            shift = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=np.float64)
            shifted = _normalize_homography(self.matrix @ shift)
            if shifted is not None:
                self.matrix = shifted
        return _transform_from_matrix(self.matrix)


def map_points(transform: ViewTransformer, points: np.ndarray) -> np.ndarray:
    result = np.full(points.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    divisor = points @ transform.m[2, :2] + transform.m[2, 2]
    valid &= np.abs(divisor) > 1e-7
    if valid.any():
        result[valid] = transform.transform_points(points[valid])
    return result


def render_minimap(base, transform: ViewTransformer, humans, colors, ball=None) -> np.ndarray:
    image = base.copy()

    def draw(detections, groups, palette, radius):
        points = map_points(transform, detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        inside = ((points >= 0) & (points <= (PITCH.length, PITCH.width))).all(axis=1)
        for index, color in enumerate(palette):
            draw_points_on_pitch(PITCH, points[inside & (groups == index)], pitch=image,
                                 face_color=sv.Color.from_hex(color), radius=radius, thickness=4)

    draw(humans, colors, COLORS, 20)
    if ball is not None and len(ball):
        draw(ball, np.zeros(len(ball), dtype=int), ("#FFFFFF",), 23)
    return image


def base_minimap() -> np.ndarray:
    return draw_pitch(PITCH)
