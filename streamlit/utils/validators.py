import os

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_FILE_SIZE_MB = 500  # optional limit

def is_allowed_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def is_under_size_limit(file_obj) -> bool:
    # file_obj may be a Streamlit UploadedFile having .size in bytes
    size_mb = getattr(file_obj, "size", 0) / (1024 * 1024)
    return size_mb <= MAX_FILE_SIZE_MB

def validate_video(file_obj) -> tuple[bool, str]:
    if not is_allowed_extension(file_obj.name):
        return False, f"Format tidak didukung. Hanya {', '.join(ALLOWED_EXTENSIONS)} yang diizinkan."
    if not is_under_size_limit(file_obj):
        return False, f"Ukuran file terlalu besar (>{MAX_FILE_SIZE_MB} MB)."
    return True, ""
