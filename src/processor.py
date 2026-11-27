from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.config import OUTPUTS, REQUIRED, TEAM_MODES, Mode
from src.core.team import prepare_team_classifier
from src.models import ModelStore
from src.modes import REGISTRY
from src.modes.context import RunContext
from src.video import encode, iter_frames, output_files, read_video_info, verify_output, video_writer


@dataclass(frozen=True)
class ProcessingResult:
    output_path: Path
    mode: Mode
    frames_processed: int
    load_seconds: float
    prepare_seconds: float | None
    process_seconds: float
    encode_seconds: float
    total_seconds: float
    warnings: tuple[str, ...]

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.process_seconds


def process_video(input_path: Path, output_path: Path | None, mode: Mode | str, device: str = "cpu",
                  progress_callback: Callable[[str, int, int], None] | None = None) -> ProcessingResult:
    start = perf_counter()
    mode = Mode(mode)
    if device != "cpu":
        raise ValueError("This application supports CPU inference only.")
    source = Path(input_path).resolve()
    target = (Path(output_path) if output_path else OUTPUTS / f"{mode.value.lower()}-{uuid4().hex}.mp4").resolve()
    if target == source:
        raise ValueError("Input and output must be different paths.")
    info = read_video_info(source)

    def notify(phase, frame=0):
        if progress_callback:
            progress_callback(phase, frame, info.frames)

    warnings = []
    prepare_seconds = None
    with output_files(target) as (intermediate, encoded):
        notify("Preparing models")
        load_start = perf_counter()
        models = ModelStore()
        for name in REQUIRED[mode]:
            models.load(name)
        load_seconds = perf_counter() - load_start
        run = RunContext(models, info.fps)
        if mode in TEAM_MODES:
            notify("Preparing team classifier")
            prepare_start = perf_counter()
            run.teams, warning = prepare_team_classifier(source, info, models)
            prepare_seconds = perf_counter() - prepare_start
            if warning:
                warnings.append(warning)

        notify("Processing frames")
        process_start = perf_counter()
        count = 0
        with closing(iter_frames(source)) as frames, closing(REGISTRY[mode](frames, run)) as output:
            with video_writer(intermediate, info) as write:
                for count, frame in enumerate(output, start=1):
                    write(frame)
                    if count == 1 or count % 5 == 0 or count == info.frames:
                        notify("Processing frames", count)
        if count == 0 or (info.frames and count != info.frames):
            raise ValueError(f"Decoded {count} frames; video reports {info.frames}. Check the source for damage.")
        process_seconds = perf_counter() - process_start

        notify("Encoding video", count)
        encode_start = perf_counter()
        encode(intermediate, encoded)
        verify_output(encoded, info, count)
        encode_seconds = perf_counter() - encode_start
    notify("Completed", count)
    warnings.extend(f"{reason} on {number} of {count} frames." for reason, number in run.warnings.items())
    return ProcessingResult(target, mode, count, load_seconds, prepare_seconds, process_seconds,
                            encode_seconds, perf_counter() - start, tuple(warnings))
