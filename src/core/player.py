import numpy as np
import supervision as sv

from src.config import GOALKEEPER, PLAYER, REFEREE, TRACK_CONFIRMATION
from src.models import ModelStore


def detect_players(frame: np.ndarray, models: ModelStore) -> sv.Detections:
    return sv.Detections.from_ultralytics(models.predict("player", frame))


def create_player_tracker(fps: float) -> sv.ByteTrack:
    return sv.ByteTrack(frame_rate=fps, minimum_consecutive_frames=TRACK_CONFIRMATION)


def track_players(tracker: sv.ByteTrack, detections: sv.Detections) -> sv.Detections:
    humans = detections[np.isin(detections.class_id, (GOALKEEPER, PLAYER, REFEREE))]
    return tracker.update_with_detections(humans)
