import os
from enum import Enum

import supervision as sv

from sports.configs.soccer import SoccerPitchConfiguration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

PLAYER_MODEL_PATH = os.path.join(DATA_DIR, 'football-player-detection.onnx')
PITCH_MODEL_PATH = os.path.join(DATA_DIR, 'football-pitch-detection.onnx')

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
ONNX_IMGSZ = 640
PITCH_CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

PALETTE = sv.ColorPalette.from_hex(COLORS)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=PALETTE, thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=PALETTE,
    text_color=sv.Color.WHITE,
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
