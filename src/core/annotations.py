import cv2
import numpy as np
import supervision as sv

from src.config import CLASS_NAMES, COLORS, GROUP_NAMES
from src.core.pitch import PITCH

PALETTE = sv.ColorPalette.from_hex(list(COLORS))
BOX = sv.BoxAnnotator(color=PALETTE)
BOX_LABEL = sv.LabelAnnotator(color=PALETTE, text_scale=0.5, text_padding=3)
ELLIPSE = sv.EllipseAnnotator(color=PALETTE, thickness=2)
ID_LABEL = sv.LabelAnnotator(color=PALETTE, text_position=sv.Position.BOTTOM_CENTER,
                             text_scale=0.45, text_padding=3)


def draw_boxes(scene, detections):
    labels = [f"{CLASS_NAMES[c]} {score:.2f}" for c, score in
              zip(detections.class_id, detections.confidence, strict=True)]
    BOX.annotate(scene, detections)
    return BOX_LABEL.annotate(scene, detections, labels=labels)


def draw_tracks(scene, humans, groups=None, show_group_names=True):
    if not len(humans):
        return scene
    labels = [f"#{track}" for track in humans.tracker_id]
    if groups is not None and show_group_names:
        labels = [f"{label} {GROUP_NAMES[group]}" for label, group in zip(labels, groups, strict=True)]
    ELLIPSE.annotate(scene, humans, custom_color_lookup=groups)
    return ID_LABEL.annotate(scene, humans, labels, custom_color_lookup=groups)


def draw_landmarks(scene, points):
    visible = {i: tuple(xy.astype(int)) for i, xy in enumerate(points) if np.isfinite(xy).all()}
    for a, b in PITCH.edges:
        if a - 1 in visible and b - 1 in visible:
            cv2.line(scene, visible[a - 1], visible[b - 1], (200, 90, 240), 2, cv2.LINE_AA)
    for index, xy in visible.items():
        cv2.circle(scene, xy, 5, (0, 220, 255), -1)
        cv2.putText(scene, PITCH.labels[index], xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (30, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(scene, PITCH.labels[index], xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return scene


def overlay_minimap(scene, radar):
    height, width = scene.shape[:2]
    scale = min(width * 0.28 / radar.shape[1], height * 0.28 / radar.shape[0])
    w = max(1, min(width, round(radar.shape[1] * scale)))
    h = max(1, min(height, round(radar.shape[0] * scale)))
    small = cv2.resize(radar, (w, h), interpolation=cv2.INTER_AREA)
    margin = min(max(12, round(min(width, height) * 0.02)), width - w, height - h)
    y2, x2 = height - margin, width - margin
    region = scene[y2 - h:y2, x2 - w:x2]
    cv2.addWeighted(small, 0.75, region, 0.25, 0, dst=region)
    return scene
