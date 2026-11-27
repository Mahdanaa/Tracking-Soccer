from collections import Counter
from dataclasses import dataclass, field

from src.core.team import TeamState
from src.models import ModelStore


@dataclass
class RunContext:
    models: ModelStore
    fps: float
    teams: TeamState | None = None
    warnings: Counter = field(default_factory=Counter)
