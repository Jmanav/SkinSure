from __future__ import annotations

import numpy as np
import torch

from api.schemas.predictions import Predictions, XAISummary
from src.explainability.gradcam import ensemble_heatmap
from src.explainability.xai_summary import summarize_heatmap
from src.preprocessing.inference_transform import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) ImageNet-normalised tensor -> (H, W, 3) uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_tensor[0].detach().cpu() * std + mean
    return (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()


class GradCamExplainer:
    """The app's explainer: Grad-CAM on the ensemble's predicted class, reduced to `XAISummary`.

    Matches the orchestrator's `Explainer` signature: (image_tensor, predictions) -> XAISummary.
    """

    def __init__(self, ensemble) -> None:
        self._ensemble = ensemble

    def __call__(self, image_tensor: torch.Tensor, predictions: Predictions) -> XAISummary:
        class_idx = self._ensemble.classes.index(predictions.primary_label.value)
        heatmap = ensemble_heatmap(self._ensemble, image_tensor, class_idx)
        return summarize_heatmap(heatmap, unnormalize(image_tensor))
