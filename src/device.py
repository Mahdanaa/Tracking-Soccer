import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class RuntimeDevice:
    kind: str
    onnx_device: str
    torch_device: str
    onnx_provider: str
    label: str
    backend: str
    strict: bool = False
    hint: str | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.onnx_provider}:{self.torch_device}"


def _has_distribution(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _detect_nvidia_gpu_name() -> str | None:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        result = subprocess.run(
            [tool, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _windowsml_active() -> bool:
    if os.name != "nt" or sys.getwindowsversion().build < 26100:
        return False
    other_builds = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn", "onnxruntime-vitisai")
    return _has_distribution("onnxruntime-windowsml") and not any(_has_distribution(b) for b in other_builds)


def _cpu_runtime(hint: str | None = None) -> RuntimeDevice:
    return RuntimeDevice(
        kind="cpu",
        onnx_device="cpu",
        torch_device="cpu",
        onnx_provider="CPUExecutionProvider",
        label="CPU",
        backend="ONNX Runtime / CPUExecutionProvider + PyTorch CPU",
        strict=False,
        hint=hint,
    )


def _cpu_hint() -> str | None:
    gpu = _detect_nvidia_gpu_name()
    if gpu:
        return (
            f"Detected {gpu}, but this environment uses CPU packages. "
            "Run `python scripts/setup_env.py` to enable GPU acceleration."
        )
    return None


def _probe_environment():
    import onnxruntime as ort
    import torch

    providers = tuple(ort.get_available_providers())
    cuda_ok = bool(torch.cuda.is_available())
    rocm_ok = cuda_ok and bool(getattr(torch.version, "hip", None))
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    migraphx_ok = importlib.util.find_spec("onnxruntime_ep_migraphx") is not None
    winml_ok = _windowsml_active()
    return providers, cuda_ok, rocm_ok, mps_ok, gpu_name, migraphx_ok, winml_ok


def resolve_device(requested: str = "auto") -> RuntimeDevice:
    choice = requested.lower()
    if choice not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device '{requested}'. Choose from: {', '.join(SUPPORTED_DEVICES)}.")
    if choice == "cpu":
        return RuntimeDevice(
            kind="cpu",
            onnx_device="cpu",
            torch_device="cpu",
            onnx_provider="CPUExecutionProvider",
            label="CPU",
            backend="ONNX Runtime / CPUExecutionProvider + PyTorch CPU",
            strict=True,
        )

    providers, cuda_ok, rocm_ok, mps_ok, gpu_name, migraphx_ok, winml_ok = _probe_environment()

    if choice == "cuda":
        if not cuda_ok:
            raise ValueError("CUDA GPU requested, but PyTorch CUDA is not available in this environment.")
        if rocm_ok:
            if not migraphx_ok:
                raise ValueError("ROCm GPU requested, but onnxruntime-ep-migraphx is not installed.")
            name = gpu_name or "AMD ROCm GPU"
            return RuntimeDevice(
                kind="rocm",
                onnx_device="0",
                torch_device="cuda",
                onnx_provider="MIGraphXExecutionProvider",
                label=f"GPU (ROCm — {name})",
                backend=f"ONNX Runtime / MIGraphXExecutionProvider + PyTorch ROCm ({name})",
                strict=True,
            )
        if "CUDAExecutionProvider" not in providers:
            raise ValueError(
                "CUDA GPU requested, but ONNX Runtime CUDAExecutionProvider is not available. "
                "Run `python scripts/setup_env.py` to install GPU dependencies."
            )
        name = gpu_name or "NVIDIA GPU"
        return RuntimeDevice(
            kind="cuda",
            onnx_device="0",
            torch_device="cuda",
            onnx_provider="CUDAExecutionProvider",
            label=f"GPU (CUDA — {name})",
            backend=f"ONNX Runtime / CUDAExecutionProvider + PyTorch CUDA ({name})",
            strict=True,
        )

    if choice == "mps":
        if not mps_ok:
            raise ValueError("Apple MPS GPU requested, but PyTorch MPS is not available on this machine.")
        has_coreml = "CoreMLExecutionProvider" in providers
        onnx_device = "mps" if has_coreml else "cpu"
        onnx_provider = "CoreMLExecutionProvider" if has_coreml else "CPUExecutionProvider"
        label = "GPU (Apple Silicon — CoreML + MPS)" if has_coreml else "GPU (Apple MPS) + CPU (ONNX)"
        return RuntimeDevice(
            kind="mps",
            onnx_device=onnx_device,
            torch_device="mps",
            onnx_provider=onnx_provider,
            label=label,
            backend=f"ONNX Runtime / {onnx_provider} + PyTorch MPS",
            strict=True,
        )

    # Auto mode: pick the fastest verified backend installed in this environment.
    if cuda_ok and rocm_ok:
        name = gpu_name or "AMD ROCm GPU"
        if migraphx_ok:
            return RuntimeDevice(
                kind="rocm",
                onnx_device="0",
                torch_device="cuda",
                onnx_provider="MIGraphXExecutionProvider",
                label=f"GPU (ROCm — {name})",
                backend=f"ONNX Runtime / MIGraphXExecutionProvider + PyTorch ROCm ({name})",
            )
        return RuntimeDevice(
            kind="rocm",
            onnx_device="cpu",
            torch_device="cuda",
            onnx_provider="CPUExecutionProvider",
            label=f"GPU (ROCm SigLIP — {name}) + CPU (ONNX)",
            backend=f"ONNX Runtime / CPUExecutionProvider + PyTorch ROCm ({name})",
        )

    if cuda_ok:
        name = gpu_name or "NVIDIA GPU"
        if "CUDAExecutionProvider" in providers:
            return RuntimeDevice(
                kind="cuda",
                onnx_device="0",
                torch_device="cuda",
                onnx_provider="CUDAExecutionProvider",
                label=f"GPU (CUDA — {name})",
                backend=f"ONNX Runtime / CUDAExecutionProvider + PyTorch CUDA ({name})",
            )
        return RuntimeDevice(
            kind="cuda",
            onnx_device="cpu",
            torch_device="cuda",
            onnx_provider="CPUExecutionProvider",
            label=f"GPU (CUDA SigLIP — {name}) + CPU (ONNX)",
            backend=f"ONNX Runtime / CPUExecutionProvider + PyTorch CUDA ({name})",
            hint="PyTorch CUDA is active, but onnxruntime-gpu is missing. Run `python scripts/setup_env.py`.",
        )

    if mps_ok:
        has_coreml = "CoreMLExecutionProvider" in providers
        onnx_device = "mps" if has_coreml else "cpu"
        onnx_provider = "CoreMLExecutionProvider" if has_coreml else "CPUExecutionProvider"
        label = "GPU (Apple Silicon — CoreML + MPS)" if has_coreml else "GPU (Apple MPS) + CPU (ONNX)"
        return RuntimeDevice(
            kind="mps",
            onnx_device=onnx_device,
            torch_device="mps",
            onnx_provider=onnx_provider,
            label=label,
            backend=f"ONNX Runtime / {onnx_provider} + PyTorch MPS",
        )

    if winml_ok:
        return RuntimeDevice(
            kind="windowsml",
            onnx_device="cpu",
            torch_device="cpu",
            onnx_provider="WindowsML",
            label="GPU/NPU (Windows ML) + CPU (SigLIP)",
            backend="ONNX Runtime / Windows ML + PyTorch CPU",
        )

    return _cpu_runtime(hint=_cpu_hint())


def preload_gpu_libraries(runtime: RuntimeDevice) -> None:
    if runtime.onnx_provider != "CUDAExecutionProvider":
        return
    import onnxruntime as ort

    # Ensure pip-installed CUDA/cuDNN shared libraries are visible on Windows as well as Linux.
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as error:
            LOGGER.warning("Could not preload ONNX Runtime CUDA libraries: %s", error)
