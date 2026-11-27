import math
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


@contextmanager
def capture(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise ValueError(f"Cannot open video: {path.name}")
        yield reader
    finally:
        reader.release()


def read_video_info(path: Path) -> VideoInfo:
    with capture(path) as reader:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video frame rate.")
        ok, first = reader.read()
        if not ok or first is None or first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("Video has no decodable color frames.")
        height, width = first.shape[:2]
        count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
        return VideoInfo(width, height, fps, max(0, int(count)) if math.isfinite(count) else 0)


def iter_frames(path: Path, stride: int = 1) -> Iterator[np.ndarray]:
    with capture(path) as reader:
        index = 0
        while reader.grab():
            if index % stride == 0:
                ok, frame = reader.retrieve()
                if not ok or frame is None:
                    raise ValueError(f"Cannot decode frame {index + 1} of {path.name}.")
                yield frame
            index += 1


def encode(source: Path, target: Path) -> None:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
               "-i", str(source), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(f"H.264 encoding failed: {result.stderr.strip()}")


def validate_frame(frame, info: VideoInfo) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or (
        frame.shape != (info.height, info.width, 3)
    ):
        raise ValueError("Expected a uint8 BGR frame at the original video dimensions.")


@contextmanager
def video_writer(path: Path, info: VideoInfo):
    size = (info.width + info.width % 2, info.height + info.height % 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), info.fps, size)
    try:
        if not writer.isOpened():
            raise OSError("Cannot create intermediate video. Check the codec, space and permissions.")

        def write(frame):
            validate_frame(frame, info)
            if size != (info.width, info.height):
                frame = cv2.copyMakeBorder(frame, 0, info.height % 2, 0, info.width % 2,
                                           cv2.BORDER_CONSTANT)
            writer.write(frame)

        yield write
    finally:
        writer.release()


@contextmanager
def output_files(target: Path):
    if target.suffix.lower() != ".mp4":
        raise ValueError("Output must have the .mp4 extension.")
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as directory:
        intermediate = Path(directory) / "frames.avi"
        encoded = Path(directory) / "encoded.mp4"
        yield intermediate, encoded
        # Publish only complete files; both operations refuse an existing destination.
        if os.name == "nt":
            encoded.rename(target)
        else:
            os.link(encoded, target)


def verify_output(path: Path, source: VideoInfo, frames: int) -> None:
    info = read_video_info(path)
    size = (source.width + source.width % 2, source.height + source.height % 2)
    if (info.width, info.height) != size or info.frames != frames or frames < 1:
        raise ValueError("Encoded dimensions or frame count do not match the processed video.")
    if abs(frames / info.fps - frames / source.fps) > 1 / source.fps:
        raise ValueError("Encoded duration differs by more than one frame.")
    with capture(path) as reader:
        codec = int(reader.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
        if codec.lower() not in {b"h264", b"avc1"}:
            raise ValueError(f"Expected H.264 output, found {codec!r}.")
