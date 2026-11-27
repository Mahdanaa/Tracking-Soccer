from src.core.ball import BallPipeline


def run_ball_detection(frames, run):
    pipeline = BallPipeline(run.models, run.fps)
    for frame in frames:
        ball = pipeline.detect(frame)
        scene = frame.copy()
        pipeline.draw(scene, ball)
        yield scene
