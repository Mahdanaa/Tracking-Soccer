import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
ACCELERATOR_PREFIXES = ("onnxruntime", "torch", "torchvision")


@dataclass(frozen=True)
class InstallPlan:
    profile: str
    reason: str
    uninstall: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]


def read_base_requirements(path: Path = REQUIREMENTS) -> list[str]:
    packages = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[\s]", line, maxsplit=1)[0].lower()
        if any(name == prefix or name.startswith(f"{prefix}-") for prefix in ACCELERATOR_PREFIXES):
            continue
        packages.append(line)
    return packages


def detect_nvidia() -> tuple[str, tuple[int, int]] | None:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        summary = subprocess.run([tool], capture_output=True, text=True, timeout=10)
        names = subprocess.run(
            [tool, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except OSError:
        return None
    if summary.returncode != 0 or names.returncode != 0:
        return None
    gpu = next((line.strip() for line in names.stdout.splitlines() if line.strip()), None)
    match = re.search(r"CUDA[^0-9]*([0-9]+)\.([0-9]+)", summary.stdout)
    if not gpu or not match:
        return None
    return gpu, (int(match.group(1)), int(match.group(2)))


def detect_rocm() -> bool:
    return sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"} and (
        Path("/dev/kfd").exists() or shutil.which("rocminfo") is not None
    )


def detect_windows_non_nvidia_gpu() -> str | None:
    if os.name != "nt" or sys.getwindowsversion().build < 26100:
        return None
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\""],
            capture_output=True, text=True, timeout=10,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        name = line.strip()
        lower = name.lower()
        if any(vendor in lower for vendor in ("amd", "radeon", "intel", "arc", "qualcomm", "adreno")):
            return name
    return None


def build_plan(mode: str = "auto", python_exe: str = sys.executable) -> InstallPlan:
    base = read_base_requirements()
    pip = (python_exe, "-m", "pip")
    selected = mode.lower()

    nvidia = detect_nvidia() if selected in {"auto", "cuda"} else None
    if selected == "cuda" and nvidia is None:
        raise RuntimeError("NVIDIA GPU with nvidia-smi not detected.")
    if nvidia is not None and sys.platform != "darwin":
        gpu_name, (major, minor) = nvidia
        if major >= 13:
            torch_index = "https://download.pytorch.org/whl/cu130"
            ort_pkg = "onnxruntime-gpu[cuda]"
        elif (major, minor) >= (12, 8):
            torch_index = "https://download.pytorch.org/whl/cu128"
            ort_pkg = "onnxruntime-gpu[cuda]<1.27"
        else:
            torch_index = "https://download.pytorch.org/whl/cu126"
            ort_pkg = "onnxruntime-gpu[cuda]<1.27"
        return InstallPlan(
            profile="cuda",
            reason=f"Detected {gpu_name} (driver CUDA {major}.{minor})",
            uninstall=("onnxruntime", "onnxruntime-windowsml", "onnxruntime-directml", "torch", "torchvision"),
            commands=(
                (*pip, "install", "torch", "torchvision", "--index-url", torch_index),
                (*pip, "install", *base, ort_pkg),
            ),
        )

    if selected == "rocm" or (selected == "auto" and detect_rocm()):
        return InstallPlan(
            profile="rocm",
            reason="Detected Linux AMD ROCm hardware",
            uninstall=("onnxruntime-gpu", "onnxruntime-windowsml", "torch", "torchvision"),
            commands=(
                (*pip, "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/rocm6.3"),
                (*pip, "install", *base, "onnxruntime"),
            ),
        )

    if selected == "auto" and sys.platform == "darwin" and platform.machine().lower() == "arm64":
        return InstallPlan(
            profile="mps",
            reason="Detected Apple Silicon (macOS arm64 — PyTorch MPS + ONNX CoreML)",
            uninstall=("onnxruntime-gpu",),
            commands=((*pip, "install", "-r", str(REQUIREMENTS)),),
        )

    win_gpu = detect_windows_non_nvidia_gpu() if selected in {"auto", "windowsml"} else None
    if selected == "windowsml" or (selected == "auto" and win_gpu is not None):
        return InstallPlan(
            profile="windowsml",
            reason=f"Detected Windows 11 24H2+ GPU ({win_gpu or 'Windows ML'})",
            uninstall=("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"),
            commands=(
                (*pip, "install", *base),
                (*pip, "install", "--upgrade", "torch", "torchvision"),
                (*pip, "install", "--upgrade", "onnxruntime-windowsml"),
            ),
        )

    return InstallPlan(
        profile="cpu",
        reason="No supported GPU accelerator detected; installing universal CPU packages",
        uninstall=("onnxruntime-gpu", "onnxruntime-windowsml", "onnxruntime-directml"),
        commands=((*pip, "install", "-r", str(REQUIREMENTS)),),
    )


def run_plan(plan: InstallPlan, python_exe: str = sys.executable, dry_run: bool = False) -> None:
    print(f"Profile: {plan.profile} ({plan.reason})")
    if plan.uninstall:
        uninstall_cmd = (python_exe, "-m", "pip", "uninstall", "-y", *plan.uninstall)
        print(">", " ".join(uninstall_cmd))
        if not dry_run:
            subprocess.run(uninstall_cmd, check=False)
    for command in plan.commands:
        print(">", " ".join(command))
        if not dry_run:
            subprocess.run(command, check=True)
    check_cmd = (python_exe, "-m", "pip", "check")
    print(">", " ".join(check_cmd))
    if not dry_run:
        subprocess.run(check_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-detect hardware and install matching CPU/GPU packages.")
    parser.add_argument("--mode", choices=("auto", "cpu", "cuda", "rocm", "windowsml"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print the detected install plan without running pip.")
    args = parser.parse_args()
    plan = build_plan(args.mode)
    run_plan(plan, dry_run=args.dry_run)
    if not args.dry_run:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.device import resolve_device

        runtime = resolve_device("auto")
        print(f"Verified runtime: {runtime.label} ({runtime.backend})")


if __name__ == "__main__":
    main()
