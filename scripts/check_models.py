import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from src.core.annotations import draw_landmarks
from src.core.pitch import PITCH, detect_pitch
from src.device import SUPPORTED_DEVICES, resolve_device
from src.models import CONTRACTS, ModelStore
from src.video import iter_frames


def main():
    parser = argparse.ArgumentParser(description="Inspect the actual ONNX sessions and run inference.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--landmarks", type=Path, help="Save a labeled PNG for visual inspection.")
    parser.add_argument("--device", choices=list(SUPPORTED_DEVICES), default="auto")
    args = parser.parse_args()
    with closing(iter_frames(args.video.resolve())) as frames:
        frame = next(frames, None)
    if frame is None:
        raise ValueError("No decodable frame in the supplied video.")
    runtime = resolve_device(args.device)
    print(f"Resolved device: {runtime.label} ({runtime.backend})")
    if runtime.hint:
        print(f"Hint: {runtime.hint}")
    models = ModelStore(runtime=runtime)
    for name in CONTRACTS:
        result = models.predict(name, frame)
        path = models.directory / f"football-{name}-detection.onnx"
        with path.open("rb") as data:
            checksum = hashlib.file_digest(data, "sha256").hexdigest()
        print(json.dumps(dict(name=path.name, sha256=checksum, boxes=len(result.boxes),
                              **models.details[name]), indent=2))
    points = detect_pitch(frame, models)
    print("Visible field labels:", [label for label, xy in zip(PITCH.labels, points)
                                    if np.isfinite(xy).all()])
    if args.landmarks:
        if args.landmarks.exists() or args.landmarks.suffix.lower() != ".png":
            raise ValueError("Choose a new .png path for the labeled frame.")
        if not cv2.imwrite(str(args.landmarks), draw_landmarks(frame.copy(), points)):
            raise OSError(f"Could not write {args.landmarks}")


if __name__ == "__main__":
    main()
