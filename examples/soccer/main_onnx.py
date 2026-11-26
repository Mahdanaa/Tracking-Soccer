import argparse
import os
from enum import Enum
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.onnx')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.onnx')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.onnx')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
ONNX_IMGSZ = 640
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2,
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2,
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


class HomographySmoother:
    """Smooths homography matrices across frames using Exponential Moving Average."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.smoothed_m: Optional[np.ndarray] = None

    def update(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if np.sum(mask) >= 4:
            try:
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32),
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


def resolve_devices(requested_device: str) -> tuple[str, str]:
    """Auto-detects GPU/CUDA availability and returns (torch_device, onnx_device)."""
    cuda_ok = torch.cuda.is_available()
    req = requested_device.lower()

    if req == 'auto':
        torch_device = 'cuda' if cuda_ok else 'cpu'
    elif req.startswith('cuda') and not cuda_ok:
        print('[WARN] CUDA diminta, tetapi GPU tidak tersedia. Beralih ke CPU.')
        torch_device = 'cpu'
    else:
        torch_device = req

    if torch_device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f'[INFO] GPU terdeteksi: {gpu_name} -> PyTorch/SigLIP pakai CUDA.')
    else:
        print('[INFO] Mode CPU aktif untuk PyTorch/SigLIP.')

    providers = ort.get_available_providers()
    if torch_device == 'cuda' and 'CUDAExecutionProvider' in providers:
        onnx_device = 'cuda'
        print('[INFO] ONNX Runtime: CUDAExecutionProvider aktif.')
    else:
        onnx_device = 'cpu'
        print('[INFO] ONNX Runtime: memakai CPUExecutionProvider.')

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

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id, dtype=int)


def predict_teams_cached(
    frame: np.ndarray,
    players: sv.Detections,
    team_classifier: TeamClassifier,
    team_cache: Dict[int, int],
    frame_idx: int,
    refresh_interval: int = 60,
) -> np.ndarray:
    """Predicts team IDs only for uncached or periodically refreshed tracker IDs."""
    if len(players) == 0:
        return np.array([], dtype=int)

    tracker_ids = players.tracker_id
    if tracker_ids is None:
        return team_classifier.predict(get_crops(frame, players)).astype(int)

    need_predict_indices = []
    for idx, tid in enumerate(tracker_ids):
        tid_int = int(tid)
        if tid_int not in team_cache or (frame_idx % refresh_interval == 0):
            need_predict_indices.append(idx)

    if need_predict_indices:
        subset = players[need_predict_indices]
        preds = team_classifier.predict(get_crops(frame, subset))
        for idx, pred in zip(need_predict_indices, preds):
            team_cache[int(tracker_ids[idx])] = int(pred)

    return np.array([team_cache[int(tid)] for tid in tracker_ids], dtype=int)


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
        pts = transformed_xy[color_lookup == class_idx]
        if len(pts) > 0:
            radar = draw_points_on_pitch(
                config=CONFIG,
                xy=pts,
                face_color=sv.Color.from_hex(COLORS[class_idx]),
                radius=20,
                pitch=radar,
            )
    return radar


def fit_team_classifier(
    player_detection_model: YOLO,
    source_video_path: str,
    device: str,
    onnx_device: str,
) -> TeamClassifier:
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device, batch_size=16)
    team_classifier.fit(crops)
    return team_classifier


def run_pitch_detection(
    source_video_path: str, onnx_device: str
) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH, task='pose')
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels
        )
        yield annotated_frame


def run_player_detection(
    source_video_path: str, onnx_device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH, task='detect')
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(
    source_video_path: str, onnx_device: str
) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH, task='detect')
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(
            image_slice, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(ONNX_IMGSZ, ONNX_IMGSZ),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(
    source_video_path: str, onnx_device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH, task='detect')
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_team_classification(
    source_video_path: str, device: str, onnx_device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH, task='detect')
    team_classifier = fit_team_classifier(
        player_detection_model, source_video_path, device, onnx_device
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    team_cache: Dict[int, int] = {}

    for frame_idx, frame in enumerate(frame_generator):
        result = player_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players_team_id = predict_teams_cached(
            frame, players, team_classifier, team_cache, frame_idx
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
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )
        yield annotated_frame


def run_radar(
    source_video_path: str, device: str, onnx_device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH, task='detect')
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH, task='pose')

    team_classifier = fit_team_classifier(
        player_detection_model, source_video_path, device, onnx_device
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    smoother = HomographySmoother(alpha=0.35)
    base_pitch = draw_pitch(config=CONFIG)
    team_cache: Dict[int, int] = {}

    for frame_idx, frame in enumerate(frame_generator):
        pitch_result = pitch_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        transformer = smoother.update(keypoints)

        player_result = player_detection_model(
            frame, imgsz=ONNX_IMGSZ, device=onnx_device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players_team_id = predict_teams_cached(
            frame, players, team_classifier, team_cache, frame_idx
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
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        h, w, _ = frame.shape
        radar = render_radar(detections, transformer, color_lookup, base_pitch)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h,
        )
        annotated_frame = sv.draw_image(
            annotated_frame, radar, opacity=0.5, rect=rect
        )
        yield annotated_frame


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    no_preview: bool,
) -> None:
    torch_device, onnx_device = resolve_devices(device)

    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(source_video_path, onnx_device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_video_path, onnx_device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(source_video_path, onnx_device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_video_path, onnx_device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path, torch_device, onnx_device
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path, torch_device, onnx_device)
    else:
        raise NotImplementedError(f'Mode {mode} is not implemented.')

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(
            frame_generator,
            total=video_info.total_frames,
            desc=f'Processing ONNX ({mode.value})',
        ):
            sink.write_frame(frame)
            if not no_preview:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if not no_preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Soccer AI runner using YOLOv8s ONNX models (imgsz=640)'
    )
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='auto (default: GPU if available else CPU), cpu, or cuda',
    )
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_TRACKING)
    parser.add_argument(
        '--no_preview',
        action='store_true',
        help='Disable live OpenCV window for faster video export',
    )
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        no_preview=args.no_preview,
    )
