from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from src.config import (GOALKEEPER, OFFICIAL, PLAYER, REFEREE, TEAM_A, TEAM_B, TEAM_MAX_CROPS,
                        TEAM_MIN_CROPS, TEAM_REFRESH, TEAM_STRIDE, UNKNOWN)
from src.core.player import detect_players
from src.models import ModelStore
from src.video import VideoInfo, iter_frames

if TYPE_CHECKING:
    from sports.common.team import TeamClassifier


def player_crops(frame, detections, indices):
    height, width = frame.shape[:2]
    for index in indices:
        box = detections.xyxy[index]
        if not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = np.clip(box, 0, (width, height, width, height)).astype(int)
        if x2 > x1 and y2 > y1:
            yield index, frame[y1:y2, x1:x2]


def prepare_team_classifier(path: Path, info: VideoInfo, models: ModelStore):
    # Reservoir sampling bounds memory while representing later camera positions too.
    stride = max(1, min(TEAM_STRIDE, info.frames // 5), info.frames // 100)
    crops = []
    seen = 0
    rng = np.random.default_rng(42)
    with closing(iter_frames(path, stride)) as frames:
        for frame in frames:
            detections = detect_players(frame, models)
            for _, crop in player_crops(frame, detections, np.flatnonzero(detections.class_id == PLAYER)):
                seen += 1
                if len(crops) < TEAM_MAX_CROPS:
                    crops.append(crop.copy())
                else:
                    slot = rng.integers(seen)
                    if slot < len(crops):
                        crops[slot] = crop.copy()
    if len(crops) < TEAM_MIN_CROPS:
        return TeamState(None), f"Only {len(crops)} player crops. At least {TEAM_MIN_CROPS} are required; teams are Unknown."

    from sports.common.team import TeamClassifier

    classifier = TeamClassifier(device="cpu", batch_size=16)
    classifier.fit(crops)
    if np.unique(classifier.cluster_model.labels_).size < 2:
        return TeamState(None), "Only one distinct cluster in the sample; teams are Unknown."
    return TeamState(classifier), None


def assign_goalkeepers(humans, groups) -> None:
    feet = humans.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    keepers = humans.class_id == GOALKEEPER
    if not keepers.any():
        return
    centroids = []
    for group in (TEAM_A, TEAM_B):
        players = feet[(humans.class_id == PLAYER) & (groups == group)]
        if not len(players):
            return
        centroids.append(players.mean(axis=0))
    distances = np.linalg.norm(feet[keepers, None] - np.asarray(centroids), axis=2)
    groups[keepers] = np.where(np.isclose(distances[:, 0], distances[:, 1]), UNKNOWN,
                               distances.argmin(axis=1))


@dataclass
class TeamState:
    classifier: "TeamClassifier | None"
    cache: dict[int, int] = field(default_factory=dict)

    def classify(self, frame, humans, frame_index) -> np.ndarray:
        groups = np.full(len(humans), UNKNOWN, dtype=int)
        if not len(humans):
            return groups
        groups[humans.class_id == REFEREE] = OFFICIAL
        if self.classifier is None:
            return groups
        ids = humans.tracker_id
        players = np.flatnonzero(humans.class_id == PLAYER)
        if frame_index % TEAM_REFRESH == 0:
            self.cache.clear()
        pending = [i for i in players if ids[i] not in self.cache]
        crops = list(player_crops(frame, humans, pending))
        if crops:
            indices, images = zip(*crops)
            predictions = self.classifier.predict(list(images))
            for index, team in zip(indices, predictions, strict=True):
                if team not in (TEAM_A, TEAM_B):
                    raise ValueError(f"Invalid team prediction: {team}")
                self.cache[int(ids[index])] = int(team)
        for index in players:
            groups[index] = self.cache.get(ids[index], UNKNOWN)
        assign_goalkeepers(humans, groups)
        return groups
