import cv2
import numpy as np
import supervision as sv

from src.config import (
    GOALKEEPER,
    ID_RECOVERY_MAX_DISTANCE,
    ID_RECOVERY_MIN_SIMILARITY,
    ID_RECOVERY_WINDOW,
    PLAYER,
    REFEREE,
    TRACK_CONFIRMATION,
    TRACK_LOST_BUFFER,
)
from src.models import ModelStore


def _torso_hsv_histogram(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.clip(box, 0, (width, height, width, height)).astype(int)
    y_mid = y1 + max(1, int((y2 - y1) * 0.55))
    crop = frame[y1:y_mid, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


class PlayerTracker:
    def __init__(self, fps: float):
        self.tracker = sv.ByteTrack(
            frame_rate=fps,
            lost_track_buffer=TRACK_LOST_BUFFER,
            minimum_consecutive_frames=TRACK_CONFIRMATION,
        )
        self.id_map: dict[int, int] = {}
        self.active: dict[int, tuple[np.ndarray, np.ndarray | None, int, int]] = {}
        self.lost: dict[int, tuple[np.ndarray, np.ndarray | None, int, int]] = {}
        self.frame_index = 0

    def _match_lost_id(
        self,
        center: np.ndarray,
        hist: np.ndarray | None,
        class_id: int,
        used_ids: set[int],
    ) -> int | None:
        if hist is None:
            return None
        best_id, best_score = None, -1.0
        for cid, (lost_center, lost_hist, lost_cls, lost_frame) in list(self.lost.items()):
            if self.frame_index - lost_frame > ID_RECOVERY_WINDOW:
                del self.lost[cid]
                continue
            if cid in used_ids or lost_cls != class_id or lost_hist is None:
                continue
            distance = float(np.linalg.norm(center - lost_center))
            if distance > ID_RECOVERY_MAX_DISTANCE:
                continue
            similarity = float(cv2.compareHist(hist, lost_hist, cv2.HISTCMP_CORREL))
            score = similarity - (distance / (ID_RECOVERY_MAX_DISTANCE * 2.0))
            if similarity >= ID_RECOVERY_MIN_SIMILARITY and score > best_score:
                best_score = score
                best_id = cid
        return best_id

    def update(self, detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        tracked = self.tracker.update_with_detections(detections)
        self.frame_index += 1
        if frame is None or tracked.tracker_id is None or not len(tracked):
            return tracked

        remapped = np.empty(len(tracked), dtype=int)
        used_ids: set[int] = set()
        for i, (box, raw_id, class_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id, tracked.class_id)):
            raw_id = int(raw_id)
            class_id = int(class_id)
            center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

            if raw_id not in self.id_map:
                hist = _torso_hsv_histogram(frame, box)
                recovered = self._match_lost_id(center, hist, class_id, used_ids)
                if recovered is not None:
                    self.id_map[raw_id] = recovered
                    self.lost.pop(recovered, None)
                else:
                    self.id_map[raw_id] = raw_id
            else:
                hist = None

            cid = self.id_map[raw_id]
            if cid in used_ids:
                cid = raw_id
                self.id_map[raw_id] = cid
            used_ids.add(cid)
            remapped[i] = cid

            if hist is None and (self.frame_index % 10 == 0 or cid not in self.active):
                hist = _torso_hsv_histogram(frame, box)
            elif hist is None:
                hist = self.active[cid][1]
            self.active[cid] = (center, hist, class_id, self.frame_index)

        for cid in list(self.active):
            if cid not in used_ids:
                self.lost[cid] = self.active.pop(cid)

        tracked.tracker_id = remapped
        return tracked


def detect_players(frame: np.ndarray, models: ModelStore) -> sv.Detections:
    return sv.Detections.from_ultralytics(models.predict("player", frame))


def create_player_tracker(fps: float) -> PlayerTracker:
    return PlayerTracker(fps=fps)


def track_players(
    tracker: PlayerTracker,
    detections: sv.Detections,
    frame: np.ndarray | None = None,
) -> sv.Detections:
    humans = detections[np.isin(detections.class_id, (GOALKEEPER, PLAYER, REFEREE))]
    return tracker.update(humans, frame=frame)
