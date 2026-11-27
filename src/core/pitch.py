import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from src.config import COLORS, HOMOGRAPHY_MODEL_TOLERANCE_PX, IMAGE_SIZE, KEYPOINT_CONFIDENCE
from src.models import ModelStore

PITCH = SoccerPitchConfiguration()
VERTICES = np.asarray(PITCH.vertices, dtype=np.float32)


def detect_pitch(frame: np.ndarray, models: ModelStore) -> np.ndarray:
    result = models.predict("pitch", frame)
    keypoints = result.keypoints
    if keypoints is None or tuple(keypoints.xy.shape[1:]) != (32, 2):
        raise ValueError("Expected pitch keypoints with shape (N,32,2).")
    points = np.full((32, 2), np.nan, dtype=np.float32)
    if not len(keypoints.xy):
        return points
    best = int(result.boxes.conf.argmax())
    xy = keypoints.xy[best].cpu().numpy()
    valid = np.isfinite(xy).all(axis=1) & (xy > 0).all(axis=1)
    if keypoints.conf is None or keypoints.conf.shape != keypoints.xy.shape[:2]:
        raise ValueError("Expected confidence for every pitch keypoint.")
    confidence = keypoints.conf[best].cpu().numpy()
    valid &= np.isfinite(confidence) & (confidence >= KEYPOINT_CONFIDENCE)
    points[valid] = xy[valid]
    return points


def build_transform(points: np.ndarray, frame_shape: tuple[int, ...]) -> ViewTransformer | None:
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
        tolerance = HOMOGRAPHY_MODEL_TOLERANCE_PX * max(frame_shape[:2]) / IMAGE_SIZE
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
