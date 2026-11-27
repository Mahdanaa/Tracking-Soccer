from src.config import Mode
from src.modes.ball_detection import run_ball_detection
from src.modes.full_match_analysis import run_full_match_analysis
from src.modes.pitch_detection import run_pitch_detection
from src.modes.player_detection import run_player_detection
from src.modes.player_tracking import run_player_tracking
from src.modes.radar import run_radar
from src.modes.team_classification import run_team_classification

REGISTRY = {
    Mode.PITCH_DETECTION: run_pitch_detection,
    Mode.PLAYER_DETECTION: run_player_detection,
    Mode.BALL_DETECTION: run_ball_detection,
    Mode.PLAYER_TRACKING: run_player_tracking,
    Mode.TEAM_CLASSIFICATION: run_team_classification,
    Mode.RADAR: run_radar,
    Mode.FULL_MATCH_ANALYSIS: run_full_match_analysis,
}
