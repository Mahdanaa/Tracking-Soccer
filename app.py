import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import streamlit as st

from src.config import OUTPUTS, SAMPLES, UPLOAD_MB, Mode
from src.device import resolve_device
from src.modes import REGISTRY
from src.processor import process_video
from src.video import read_video_info

DESCRIPTIONS = {
    Mode.PITCH_DETECTION: "Mark visible field landmarks and the lines between them.",
    Mode.PLAYER_DETECTION: "Show boxes and confidence for players, goalkeepers, referees and the ball.",
    Mode.BALL_DETECTION: "Mark the ball and its recent detected positions.",
    Mode.PLAYER_TRACKING: "Follow players, goalkeepers and referees with temporary track IDs.",
    Mode.TEAM_CLASSIFICATION: "Color tracked players by jersey group and identify referees.",
    Mode.RADAR: "Project tracked people onto a two-dimensional pitch.",
    Mode.FULL_MATCH_ANALYSIS: "Combine team tracking and radar with a dedicated ball detector and trail.",
}


@st.cache_resource
def detected_runtime():
    return resolve_device("auto")


def label(mode):
    return mode.value.replace("_", " ").title()


def select_source(directory):
    samples = sorted(SAMPLES.glob("*.mp4"))
    source = st.radio("Video source", ["Upload", "Sample"] if samples else ["Upload"], horizontal=True)
    uploaded = None
    if source == "Upload":
        uploaded = st.file_uploader(f"Upload MP4 (up to {UPLOAD_MB} MB)", type=["mp4"],
                                    max_upload_size=UPLOAD_MB)
    previous = st.session_state.get("upload")
    if previous and (uploaded is None or uploaded.file_id != previous[0]):
        previous[1].unlink(missing_ok=True)
        del st.session_state.upload
    if source == "Sample":
        sample = st.selectbox("Sample video", samples, format_func=lambda path: path.name)
        return sample, sample.name
    if uploaded is None:
        return None, None
    if "upload" not in st.session_state:
        path = directory / f"input-{uuid4().hex}.mp4"
        path.write_bytes(uploaded.getbuffer())
        st.session_state.upload = (uploaded.file_id, path)
    return st.session_state.upload[1], Path(uploaded.name).name


def main():
    st.set_page_config(page_title="Football Player Tracking & Analysis", layout="wide")
    st.title("Football Player Tracking & Analysis")
    st.write("Analyze recorded football footage locally with computer vision. Result videos have no audio.")
    OUTPUTS.mkdir(exist_ok=True)
    if "directory" not in st.session_state:
        st.session_state.directory = TemporaryDirectory(prefix="session-", dir=OUTPUTS)
    directory = Path(st.session_state.directory.name)

    left, right = st.columns(2)
    with left:
        path, name = select_source(directory)
        info = None
        if path:
            try:
                info = read_video_info(path)
                duration = f"{info.frames / info.fps:.2f} s" if info.frames else "Duration unknown"
                st.caption(f"{name} · {info.width} × {info.height} · {info.fps:g} FPS · {duration}")
            except (OSError, ValueError) as error:
                st.error(str(error))
    runtime = detected_runtime()
    with right:
        modes = list(REGISTRY)
        mode = st.selectbox("Analysis mode", modes, index=modes.index(Mode.PLAYER_TRACKING), format_func=label)
        st.write(DESCRIPTIONS[mode])
        st.caption(f"Device · Auto ({runtime.label})")
        if runtime.hint:
            st.caption(runtime.hint)
        start = st.button("Start Analysis", type="primary", disabled=info is None)

    key = (path, path.stat().st_mtime_ns if info else None, mode, runtime.key)
    if "result" in st.session_state and (st.session_state.result[0] != key or start):
        st.session_state.result[1].output_path.unlink(missing_ok=True)
        del st.session_state.result
    if start:
        status = st.empty()
        progress = st.empty()

        def update(phase, done, total):
            status.write(phase)
            if phase == "Processing frames" and total:
                progress.progress(min(1.0, done / total), text=f"{done} / {total} frames")
            else:
                progress.empty()

        try:
            target = directory / f"{mode.value.lower()}-{uuid4().hex}.mp4"
            result = process_video(path, target, mode, device="auto", progress_callback=update)
            st.session_state.result = (key, result)
        except Exception as error:
            logging.exception("Analysis failed")
            progress.empty()
            status.error(f"Failed — {error}")

    original, processed = st.columns(2)
    if info:
        original.subheader("Original video")
        original.video(str(path))
        original.caption("If the original codec cannot play here, the source metadata is shown above.")
    if "result" in st.session_state:
        result = st.session_state.result[1]
        with processed:
            st.subheader(f"{label(result.mode)} result")
            st.video(str(result.output_path))
            st.write(f"{result.frames_processed} frames · {result.total_seconds:.2f} s total · "
                     f"{result.processing_fps:.2f} processing FPS · {result.device_label}")
            for warning in result.warnings:
                st.warning(warning)
            stem = re.sub(r"[^\w.-]", "_", Path(name).stem)
            with result.output_path.open("rb") as video:
                st.download_button("Download result", video, f"{stem}_{mode.value.lower()}.mp4",
                                   mime="video/mp4", on_click="ignore")


if __name__ == "__main__":
    main()
