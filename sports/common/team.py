from functools import lru_cache
from itertools import batched

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, SiglipVisionModel

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


@lru_cache(maxsize=2)
def _load_siglip(device: str):
    try:
        processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_PATH, local_files_only=True)
        model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH, local_files_only=True
        ).to(device).eval()
        return processor, model
    except OSError as error:
        raise RuntimeError(f"SigLIP cache incomplete. Run the README setup for {SIGLIP_MODEL_PATH}.") from error


class TeamClassifier:
    def __init__(self, device: str = "cpu", batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size
        self.processor, self.features_model = _load_siglip(device)
        self.reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)

    @torch.inference_mode()
    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        features = []
        for batch in batched(crops, self.batch_size):
            images = [sv.cv2_to_pillow(crop) for crop in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.features_model(**inputs)
            features.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.concatenate(features)

    def fit(self, crops: list[np.ndarray]) -> None:
        projections = self.reducer.fit_transform(self.extract_features(crops))
        self.cluster_model.fit(projections)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty(0, dtype=int)
        projections = self.reducer.transform(self.extract_features(crops))
        return self.cluster_model.predict(projections)
