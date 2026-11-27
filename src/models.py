import ast
from pathlib import Path

import numpy as np

from src.config import BALL_CONFIDENCE, CLASS_NAMES, CONFIDENCE, IMAGE_SIZE, MODEL_DIR, MODEL_IOU
from src.device import RuntimeDevice, preload_gpu_libraries, resolve_device

CONTRACTS = {
    "player": ("detect", CLASS_NAMES, 8),
    "ball": ("detect", {0: "ball"}, 5),
    "pitch": ("pose", {0: "pitch"}, 101),
}


def validate_session(name, session, runtime: RuntimeDevice | None = None) -> dict:
    task, names, channels = CONTRACTS[name]
    inputs, outputs = session.get_inputs(), session.get_outputs()
    metadata = session.get_modelmeta().custom_metadata_map
    if len(inputs) != 1 or inputs[0].shape != [1, 3, IMAGE_SIZE, IMAGE_SIZE]:
        raise ValueError(f"{name}: expected static float32 input [1,3,640,640].")
    if len(outputs) != 1 or outputs[0].shape != [1, channels, 8400]:
        raise ValueError(f"{name}: incompatible output shape, expected [1,{channels},8400].")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError(f"{name}: expected float32 tensors.")
    if metadata.get("task") != task or ast.literal_eval(metadata.get("names", "{}")) != names:
        raise ValueError(f"{name}: expected task {task} and class mapping {names}.")
    if name == "pitch" and ast.literal_eval(metadata.get("kpt_shape", "[]")) != [32, 3]:
        raise ValueError("Pitch model must contain 32 field keypoints with confidence.")
    if name == "pitch" and "kpt_names" in metadata:
        expected_order = {0: [str(index) for index in range(32)]}
        if ast.literal_eval(metadata["kpt_names"]) != expected_order:
            raise ValueError("Pitch keypoint order differs from the verified team export.")
    providers = session.get_providers()
    expected = runtime.onnx_provider if runtime else "CPUExecutionProvider"
    strict = runtime.strict if runtime else True
    warning = None
    if not providers:
        raise RuntimeError(f"{name}: ONNX Runtime session has no active providers.")
    if expected == "CPUExecutionProvider":
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected inference providers: {providers}")
    elif expected != "WindowsML" and expected not in providers:
        if strict:
            raise RuntimeError(f"Requested {expected}, but ONNX Runtime fell back to {providers}.")
        warning = f"ONNX {name} fell back to {providers[0]} instead of {expected}."
    return {"input": inputs[0].shape, "output": outputs[0].shape, "dtype": inputs[0].type,
            "providers": providers, "metadata": metadata, "warning": warning}


class ModelStore:
    def __init__(self, directory: Path = MODEL_DIR, runtime: RuntimeDevice | None = None):
        self.directory = directory.resolve()
        self.runtime = runtime or resolve_device("auto")
        self.predictors = {}
        self.details = {}
        self.warnings = []
        self._preloaded = False

    def load(self, name: str):
        if name in self.predictors:
            return self.predictors[name]
        path = self.directory / f"football-{name}-detection.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Place the original {name} model at {path}")
        if not self._preloaded:
            preload_gpu_libraries(self.runtime)
            self._preloaded = True
        from ultralytics.models.yolo.detect import DetectionPredictor
        from ultralytics.models.yolo.pose import PosePredictor

        task = CONTRACTS[name][0]
        predictor_type = PosePredictor if task == "pose" else DetectionPredictor
        confidence = BALL_CONFIDENCE if name == "ball" else CONFIDENCE
        options = dict(task=task, model=str(path), device=self.runtime.onnx_device, imgsz=IMAGE_SIZE, batch=1,
                       rect=False, conf=confidence, iou=MODEL_IOU, verbose=False, save=False)
        predictor = predictor_type(overrides=options)
        predictor.setup_model(str(path), verbose=False)
        details = validate_session(name, predictor.model.session, self.runtime)
        if details["warning"]:
            self.warnings.append(details["warning"])
        self.details[name] = details
        self.predictors[name] = predictor
        return predictor

    def active_provider(self) -> str:
        providers = [info["providers"][0] for info in self.details.values() if info.get("providers")]
        return providers[0] if providers else self.runtime.onnx_provider

    def predict(self, name: str, frame: np.ndarray):
        return self.load(name)(source=frame)[0]
