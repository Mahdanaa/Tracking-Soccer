import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("supervision", "supervision"),
    ("ultralytics", "ultralytics"),
    ("transformers", "transformers"),
    ("umap", "umap-learn"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
    ("gdown", "gdown"),
    ("sports", "sports (pip install -e .)"),
]

REQUIRED_MODELS = [
    "football-player-detection.pt",
    "football-pitch-detection.pt",
    "football-ball-detection.pt",
]


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> bool:
    print_header("1. Python")
    ver = sys.version_info
    ok = ver >= (3, 8)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {ver.major}.{ver.minor}.{ver.micro} (minimal 3.8)")
    return ok


def check_nvidia_smi() -> bool:
    print_header("2. Driver GPU & nvidia-smi")
    if shutil.which("nvidia-smi") is None:
        print("[FAIL] nvidia-smi tidak ditemukan. Pastikan driver NVIDIA terinstal.")
        return False

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[OK] GPU terdeteksi: {res.stdout.strip()}")
        return True
    except Exception as exc:
        print(f"[FAIL] Gagal menjalankan nvidia-smi: {exc}")
        return False


def check_packages() -> bool:
    print_header("3. Paket Python")
    all_ok = True
    for module_name, label in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[OK]   {label:<28} ({ver})")
        except Exception:
            print(f"[MISS] {label:<28} -> belum terpasang")
            all_ok = False
    return all_ok


def check_pytorch_cuda() -> bool:
    print_header("4. PyTorch CUDA (RTX 2050)")
    try:
        import torch
    except ImportError:
        print("[FAIL] torch belum terpasang.")
        return False

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA built ver  : {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() = False (masih PyTorch CPU-only!)")
        print("       Solusi: instal PyTorch CUDA, misal:")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK]   CUDA aktif! Device: {gpu_name} | VRAM: {vram_gb:.2f} GB")
    if vram_gb < 5.0:
        print("[INFO] VRAM ~4 GB: cukup untuk semua mode, namun pada mode RADAR")
        print("       (2 YOLO + SigLIP), pertimbangkan kecilkan batch_size TeamClassifier jika OOM.")
    return True


def check_data_files() -> bool:
    print_header("5. Model & Video Sampel (examples/soccer/data)")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "examples" / "soccer" / "data"

    if not data_dir.exists():
        print(f"[MISS] Folder tidak ditemukan: {data_dir}")
        return False

    all_ok = True
    for model_name in REQUIRED_MODELS:
        model_path = data_dir / model_name
        if model_path.exists() and model_path.stat().st_size > 0:
            size_mb = model_path.stat().st_size / (1024**2)
            print(f"[OK]   {model_name} ({size_mb:.1f} MB)")
        else:
            print(f"[MISS] {model_name}")
            all_ok = False

    videos = list(data_dir.glob("*.mp4"))
    if videos:
        print(f"[OK]   Video sampel ditemukan ({len(videos)} file): {', '.join(v.name for v in videos)}")
    else:
        print("[MISS] Belum ada file .mp4 di examples/soccer/data/")
        all_ok = False

    return all_ok


def main() -> None:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    results = [
        check_python(),
        check_nvidia_smi(),
        check_packages(),
        check_pytorch_cuda(),
        check_data_files(),
    ]

    print_header("Ringkasan")
    if all(results):
        print("[SIAP] Semua kebutuhan terpenuhi. Anda bisa langsung menjalankan dengan --device cuda")
    else:
        print("[BELUM SIAP] Masih ada komponen [MISS]/[FAIL] di atas yang perlu dilengkapi.")


if __name__ == "__main__":
    main()
