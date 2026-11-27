from src.modes.radar import run_radar


def run_full_match_analysis(frames, run):
    yield from run_radar(frames, run, include_ball=True)
