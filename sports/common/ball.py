from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


@dataclass(frozen=True)
class BallCandidates:
    points: np.ndarray
    scores: np.ndarray
    contrast: np.ndarray


class BallTracker:
    def __init__(self, fps: float = 25.0, acquisition_confidence: float = 0.25):
        self.time_scale = float(np.clip(25 / fps, 0.5, 2.0))
        self.max_misses = max(3, round(fps * 0.24))
        self.acquisition_confidence = acquisition_confidence
        self.history = deque(maxlen=2)
        self.camera_offset = np.zeros(2)
        self.position = None
        self.velocity = np.zeros(2)
        self.misses = 0

    def reset(self) -> None:
        self.history.clear()
        self.camera_offset.fill(0)
        self.position = None
        self.velocity.fill(0)
        self.misses = 0

    def acquire(self, current: BallCandidates) -> tuple[int, np.ndarray] | None:
        weights = current.scores + 0.4 * current.contrast
        index = self.choose(np.where(current.scores >= self.acquisition_confidence, weights, -np.inf),
                            current.points, min_margin=0.12)
        if index is not None and current.scores[index] >= 0.55 and current.contrast[index] >= 0.35:
            if self.history:
                previous = self.history[-1]
                offsets = current.points[index] - previous.points
                distances = np.linalg.norm(offsets, axis=1)
                nearby = np.flatnonzero((distances <= 28 * self.time_scale) & (previous.scores >= 0.45))
                if len(nearby):
                    return index, offsets[nearby[distances[nearby].argmin()]]
            if not self.history and current.scores[index] >= 0.74:
                return index, np.zeros(2)
        if len(self.history) < 2:
            return None
        older, previous = self.history
        if not len(older.points) or not len(previous.points):
            return None
        scores = np.full(len(current.points), -np.inf)
        velocities = np.zeros_like(current.points)
        credible = np.flatnonzero(current.scores >= self.acquisition_confidence)
        crowded = len(credible) > 1
        for index in credible:
            delta = current.points[index] - previous.points
            earlier = previous.points - delta
            errors = np.linalg.norm(earlier[:, None] - older.points, axis=2) / self.time_scale
            distances = np.linalg.norm(delta, axis=1) / self.time_scale
            confidence = (current.scores[index] + previous.scores[:, None] + older.scores) / 3
            valid = (distances[:, None] <= 55) & (errors <= 18) & (confidence >= 0.35)
            motion = np.linalg.norm(current.points[index] - older.points, axis=1) / (2 * self.time_scale)
            if crowded:
                # Camera-relative motion helps reject persistent markings during acquisition.
                valid &= motion >= 2.5
            appearance = (current.contrast[index] + previous.contrast[:, None] + older.contrast) / 3
            weights = confidence + 0.37 * appearance + 0.25 * np.minimum(motion / 10, 1) - 0.02 * errors
            weights[~valid] = -np.inf
            previous_index, older_index = np.unravel_index(weights.argmax(), weights.shape)
            scores[index] = weights[previous_index, older_index]
            velocities[index] = (current.points[index] - older.points[older_index]) / 2

        choice = self.choose(scores, current.points, min_margin=0.10)
        return (choice, velocities[choice]) if choice is not None else None

    @staticmethod
    def choose(scores: np.ndarray, points: np.ndarray, min_margin: float) -> int | None:
        if not len(scores):
            return None
        index = int(scores.argmax())
        if not np.isfinite(scores[index]):
            return None
        separate = np.linalg.norm(points - points[index], axis=1) > 8
        if np.any(scores[separate] > scores[index] - min_margin):
            return None
        return index

    def follow(self, current: BallCandidates) -> tuple[int, np.ndarray] | None:
        gap = self.misses + 1
        predicted = self.position + self.velocity * gap
        error = np.linalg.norm(current.points - predicted, axis=1)
        radius = min(64, 23 + 7 * (gap - 1)) * self.time_scale + 0.35 * np.linalg.norm(self.velocity)
        # A low-confidence observation can only continue a track close to its prediction.
        weak = current.scores < self.acquisition_confidence
        valid = (error <= radius) & (~weak | (error <= 0.5 * radius))
        scores = np.where(valid, current.scores + 0.4 * current.contrast - 0.65 * error / radius, -np.inf)
        choice = self.choose(scores, current.points, min_margin=0.08)
        if choice is None:
            return None
        velocity = (current.points[choice] - self.position) / gap
        return choice, 0.25 * self.velocity + 0.75 * velocity

    def update(self, detections: sv.Detections, camera_shift=(0, 0), scale: float = 1.0,
               contrast: np.ndarray | None = None) -> sv.Detections:
        self.camera_offset += np.asarray(camera_shift) * scale
        points = detections.get_anchors_coordinates(sv.Position.CENTER) * scale - self.camera_offset
        scores = detections.confidence if detections.confidence is not None else np.zeros(len(points))
        current = BallCandidates(points, scores, contrast if contrast is not None else np.zeros(len(points)))
        match = self.follow(current) if self.position is not None else None
        if match is None and (self.position is None or self.misses >= 1):
            match = self.acquire(current)
        self.history.append(current)
        if match is None:
            if self.position is not None:
                self.misses += 1
                if self.misses >= self.max_misses:
                    self.position = None
                    self.misses = 0
            return detections[:0]
        index, self.velocity = match
        self.position = points[index]
        self.misses = 0
        return detections[[index]]
