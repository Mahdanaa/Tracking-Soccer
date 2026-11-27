from src.core.annotations import draw_tracks, overlay_minimap
from src.core.ball import BallPipeline
from src.core.pitch import base_minimap, build_transform, detect_pitch, render_minimap
from src.modes.team_classification import tracked_teams


def run_radar(frames, run, include_ball=False):
    pipeline = BallPipeline(run.models, run.fps) if include_ball else None
    base = base_minimap()
    for frame, humans, groups in tracked_teams(frames, run):
        transform = build_transform(detect_pitch(frame, run.models), frame.shape)
        ball = pipeline.detect(frame) if pipeline else None
        scene = draw_tracks(frame.copy(), humans, groups, show_group_names=False)
        if pipeline:
            pipeline.draw(scene, ball)
        if transform is None:
            run.warnings["Minimap unavailable"] += 1
        else:
            minimap = render_minimap(base, transform, humans, groups, ball)
            scene = overlay_minimap(scene, minimap)
        yield scene
