from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ALLOWED_VIDEO_EXTENSIONS, UPLOAD_MB  # noqa: E402


def is_allowed_extension(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_VIDEO_EXTENSIONS


def is_under_size_limit(file_obj) -> bool:
    size_mb = getattr(file_obj, "size", 0) / (1024 * 1024)
    return size_mb <= UPLOAD_MB


def validate_video(file_obj) -> tuple[bool, str]:
    if not is_allowed_extension(file_obj.name):
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        return False, f"Format tidak didukung. Hanya {allowed} yang diizinkan."
    if not is_under_size_limit(file_obj):
        return False, f"Ukuran file terlalu besar (>{UPLOAD_MB} MB)."
    return True, ""
