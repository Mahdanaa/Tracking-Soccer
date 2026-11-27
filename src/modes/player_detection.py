from src.core.annotations import draw_boxes
from src.core.player import detect_players


def run_player_detection(frames, run):
    for frame in frames:
        yield draw_boxes(frame.copy(), detect_players(frame, run.models))
