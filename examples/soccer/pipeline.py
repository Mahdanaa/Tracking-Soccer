from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import (
    COLORS,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    ONNX_IMGSZ,
    PITCH_CONFIG,
    PITCH_MODEL_PATH,
    PLAYER_CLASS_ID,
    PLAYER_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
    Mode,
)
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer


class HomographySmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.smoothed_m: Optional[np.ndarray] = None

    def update(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if np.sum(mask) >= 4:
            try:
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(PITCH_CONFIG.vertices)[mask].astype(np.float32),
                )
                if self.smoothed_m is None:
                    self.smoothed_m = transformer.m.copy()
                else:
                    self.smoothed_m = (
                        self.alpha * transformer.m + (1.0 - self.alpha) * self.smoothed_m
                    )
                transformer.m = self.smoothed_m
                return transformer
            except ValueError:
                pass

        if self.smoothed_m is None:
            return None

        fallback = ViewTransformer.__new__(ViewTransformer)
        fallback.m = self.smoothed_m
        return fallback


def resolve_devices(requested_device: str) -> Tuple[str, str]:
    cuda_available = torch.cuda.is_available()
    target = requested_device.lower()

    if target == 'auto':
        torch_device = 'cuda' if cuda_available else 'cpu'
    elif target.startswith('cuda') and not cuda_available:
        print('[WARN] CUDA requested but unavailable. Falling back to CPU.')
        torch_device = 'cpu'
    else:
        torch_device = target

    providers = ort.get_available_providers()
    onnx_device = (
        'cuda' if torch_device == 'cuda' and 'CUDAExecutionProvider' in providers else 'cpu'
    )
    print(f'[INFO] Active devices -> PyTorch: {torch_device.upper()} | ONNX: {onnx_device.upper()}')
    return torch_device, onnx_device


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    team_0_mask = players_team_id == 0
    team_1_mask = players_team_id == 1
    if not np.any(team_0_mask) or not np.any(team_1_mask):
        default_team = 0 if np.any(team_0_mask) else 1
        return np.full(len(goalkeepers), default_team, dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)

    return np.array([
        0 if np.linalg.norm(xy - team_0_centroid) < np.linalg.norm(xy - team_1_centroid) else 1
        for xy in goalkeepers_xy
    ], dtype=int)


def predict_teams_cached(
    frame: np.ndarray,
    players: sv.Detections,
    classifier: TeamClassifier,
    cache: Dict[int, int],
    frame_idx: int,
    refresh_interval: int = 60,
) -> np.ndarray:
    if len(players) == 0:
        return np.array([], dtype=int)

    if players.tracker_id is None:
        return classifier.predict(get_crops(frame, players)).astype(int)

    indices_to_predict = [
        i for i, tid in enumerate(players.tracker_id)
        if int(tid) not in cache or frame_idx % refresh_interval == 0
    ]

    if indices_to_predict:
        predictions = classifier.predict(get_crops(frame, players[indices_to_predict]))
        for idx, team_id in zip(indices_to_predict, predictions):
            cache[int(players.tracker_id[idx])] = int(team_id)

    return np.array([cache[int(tid)] for tid in players.tracker_id], dtype=int)


def fit_team_classifier(
    player_model: YOLO,
    source_video_path: str,
    torch_device: str,
    onnx_device: str,
) -> TeamClassifier:
    crops: List[np.ndarray] = []
    for frame in tqdm(
        sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE),
        desc='Collecting player crops',
    ):
        result = player_model(frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    classifier = TeamClassifier(device=torch_device, batch_size=16)
    classifier.fit(crops)
    return classifier


def render_radar(
    detections: sv.Detections,
    transformer: Optional[ViewTransformer],
    color_lookup: np.ndarray,
    base_pitch: np.ndarray,
) -> np.ndarray:
    radar = base_pitch.copy()
    if transformer is None or len(detections) == 0:
        return radar

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    for class_idx in range(4):
        points = transformed_xy[color_lookup == class_idx]
        if len(points) > 0:
            radar = draw_points_on_pitch(
                config=PITCH_CONFIG,
                xy=points,
                face_color=sv.Color.from_hex(COLORS[class_idx]),
                radius=20,
                pitch=radar,
            )
    return radar


def overlay_radar(frame: np.ndarray, radar: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    resized_radar = sv.resize_image(radar, (w // 2, h // 2))
    radar_h, radar_w, _ = resized_radar.shape
    rect = sv.Rect(
        x=w // 2 - radar_w // 2,
        y=h - radar_h,
        width=radar_w,
        height=radar_h,
    )
    return sv.draw_image(frame, resized_radar, opacity=0.5, rect=rect)


def analyze_video(
    source_video_path: str,
    mode: Mode = Mode.RADAR,
    device: str = 'auto',
) -> Iterator[np.ndarray]:
    torch_device, onnx_device = resolve_devices(device)
    player_model = YOLO(PLAYER_MODEL_PATH, task='detect')
    classifier = fit_team_classifier(
        player_model, source_video_path, torch_device, onnx_device
    )

    use_radar = mode == Mode.RADAR
    pitch_model = YOLO(PITCH_MODEL_PATH, task='pose') if use_radar else None
    smoother = HomographySmoother() if use_radar else None
    base_pitch = draw_pitch(config=PITCH_CONFIG) if use_radar else None

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    team_cache: Dict[int, int] = {}

    for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path=source_video_path)):
        player_result = player_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players_team_id = predict_teams_cached(
            frame, players, classifier, team_cache, frame_idx
        )
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tid) for tid in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        if use_radar and pitch_model is not None and smoother is not None and base_pitch is not None:
            pitch_result = pitch_model(
                frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
            )[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            transformer = smoother.update(keypoints)
            radar = render_radar(detections, transformer, color_lookup, base_pitch)
            annotated_frame = overlay_radar(annotated_frame, radar)

        yield annotated_frame
