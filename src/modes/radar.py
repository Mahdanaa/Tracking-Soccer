from src.config import PITCH_STARTUP_BUFFER
from src.core.annotations import draw_tracks, overlay_minimap
from src.core.ball import BallPipeline
from src.core.pitch import (PitchStabilizer, base_minimap, detect_pitch_transform, render_minimap)
from src.modes.team_classification import tracked_teams


def _compose(scene, base, transform, humans, groups, ball):
    minimap = render_minimap(base, transform, humans, groups, ball) if transform is not None else base
    return overlay_minimap(scene, minimap)


def run_radar(frames, run, include_ball=False):
    pipeline = BallPipeline(run.models, run.fps) if include_ball else None
    stabilizer = PitchStabilizer()
    base = base_minimap()
    pending = []
    started = False
    for frame, humans, groups in tracked_teams(frames, run):
        transform = stabilizer.update(frame, detect_pitch_transform(frame, run.models))
        ball = pipeline.detect(frame) if pipeline else None
        scene = draw_tracks(frame.copy(), humans, groups, show_group_names=False)
        if pipeline:
            pipeline.draw(scene, ball)
        if not started and transform is None and len(pending) < PITCH_STARTUP_BUFFER:
            pending.append((scene, humans, groups, ball))
            continue
        started = True
        if pending:
            for item in pending:
                yield _compose(item[0], base, transform, item[1], item[2], item[3])
            pending.clear()
        yield _compose(scene, base, transform, humans, groups, ball)
    for item in pending:
        yield _compose(item[0], base, None, item[1], item[2], item[3])
