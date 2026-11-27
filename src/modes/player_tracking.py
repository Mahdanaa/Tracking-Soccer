from src.core.annotations import draw_tracks
from src.core.player import create_player_tracker, detect_players, track_players


def run_player_tracking(frames, run):
    tracker = create_player_tracker(run.fps)
    for frame in frames:
        humans = track_players(tracker, detect_players(frame, run.models))
        yield draw_tracks(frame.copy(), humans)
