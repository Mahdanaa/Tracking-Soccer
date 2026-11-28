from src.core.annotations import draw_tracks
from src.core.player import create_player_tracker, detect_players, track_players


def tracked_teams(frames, run):
    tracker = create_player_tracker(run.fps)
    for index, frame in enumerate(frames):
        humans = track_players(tracker, detect_players(frame, run.models), frame=frame)
        groups = run.teams.classify(frame, humans, index)
        yield frame, humans, groups


def run_team_classification(frames, run):
    for frame, humans, groups in tracked_teams(frames, run):
        yield draw_tracks(frame.copy(), humans, groups)
