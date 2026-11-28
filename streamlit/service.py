from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Callable, Optional
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_MODE,
    HISTORY_FILE,
    OUTPUTS,
    UPLOADS_DIR,
)
from src.device import resolve_device  # noqa: E402
from src.processor import process_video  # noqa: E402
from src.video import read_video_info  # noqa: E402


class AnalysisError(Exception):
    pass


def _ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _load_history() -> list[dict]:
    _ensure_dirs()
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_history(records: list[dict]) -> None:
    _ensure_dirs()
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _find_record(analysis_id: str) -> tuple[list[dict], Optional[dict]]:
    records = _load_history()
    for item in records:
        if item.get("analysis_id") == analysis_id:
            return records, item
    return records, None


def get_engine_status() -> tuple[bool, str]:
    try:
        runtime = resolve_device(DEFAULT_DEVICE)
        return True, f"Engine lokal siap ({runtime.label})"
    except Exception as exc:
        return False, f"Engine lokal bermasalah: {exc}"


def submit_video(file_obj) -> dict:
    _ensure_dirs()
    analysis_id = uuid4().hex[:12]
    original_name = Path(getattr(file_obj, "name", "video.mp4")).name
    suffix = Path(original_name).suffix or ".mp4"
    input_path = UPLOADS_DIR / f"{analysis_id}{suffix}"

    try:
        file_obj.seek(0)
        input_path.write_bytes(file_obj.read())
    except Exception as exc:
        raise AnalysisError(f"Gagal menyimpan file video: {exc}") from exc

    duration = None
    try:
        info = read_video_info(input_path)
        if info.frames and info.fps:
            duration = round(info.frames / info.fps, 2)
    except Exception:
        duration = None

    records = _load_history()
    records.insert(
        0,
        {
            "analysis_id": analysis_id,
            "filename": original_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "queued",
            "progress": 0,
            "input_path": str(input_path),
            "video_path": None,
            "error": None,
            "summary": {"duration": duration} if duration is not None else {},
        },
    )
    _save_history(records)
    return {"analysis_id": analysis_id}


def run_analysis(
    analysis_id: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    records, record = _find_record(analysis_id)
    if record is None:
        raise AnalysisError(f"Data analisis `{analysis_id}` tidak ditemukan.")

    input_path = Path(record["input_path"])
    if not input_path.exists():
        record["status"] = "failed"
        record["error"] = "File video input tidak ditemukan."
        _save_history(records)
        raise AnalysisError(record["error"])

    record["status"] = "processing"
    _save_history(records)

    output_path = OUTPUTS / f"{analysis_id}_annotated.mp4"
    try:
        result = process_video(
            input_path=input_path,
            output_path=output_path,
            mode=DEFAULT_MODE,
            device=DEFAULT_DEVICE,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        _save_history(records)
        raise AnalysisError(f"Analisis gagal: {exc}") from exc

    summary = dict(record.get("summary") or {})
    summary.update(
        {
            "frames_processed": result.frames_processed,
            "processing_fps": round(result.processing_fps, 2),
            "total_seconds": round(result.total_seconds, 2),
            "device": result.device_label,
        }
    )

    record["status"] = "completed"
    record["progress"] = 100
    record["video_path"] = str(result.output_path)
    record["error"] = None
    record["summary"] = summary
    _save_history(records)
    return record


def get_status(analysis_id: str) -> dict:
    _, record = _find_record(analysis_id)
    if record is None:
        raise AnalysisError(f"ID analisis `{analysis_id}` tidak ditemukan.")
    return {
        "analysis_id": analysis_id,
        "status": record.get("status", "unknown"),
        "progress": record.get("progress", 0),
        "error": record.get("error"),
    }


def get_results(analysis_id: str) -> dict:
    _, record = _find_record(analysis_id)
    if record is None:
        raise AnalysisError(f"Hasil untuk `{analysis_id}` tidak ditemukan.")
    return {
        "analysis_id": analysis_id,
        "status": record.get("status"),
        "video_path": record.get("video_path") or record.get("video_url"),
        "summary": record.get("summary") or {},
    }


def get_history() -> dict:
    return {"analyses": _load_history()}


def download_file(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists():
        raise AnalysisError(f"File hasil tidak ditemukan: {path.name}")
    try:
        return path.read_bytes()
    except Exception as exc:
        raise AnalysisError(f"Gagal membaca file hasil: {exc}") from exc
