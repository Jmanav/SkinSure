from __future__ import annotations

from typing import Callable

import numpy as np
import torch

DELETION_PERCENTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

ProbsFn = Callable[[torch.Tensor], np.ndarray]   # (1, 3, H, W) -> (num_classes,) probabilities


def ensemble_probs_fn(ensemble) -> ProbsFn:
    """Weighted softmax average of the ensemble's backbones (what SkinEnsemble.predict uses)."""
    @torch.no_grad()
    def probs(image_tensor: torch.Tensor) -> np.ndarray:
        x = image_tensor.to(ensemble.device)
        total = 0.0
        for model, weight in zip(ensemble.backbones.values(), ensemble.weights):
            total = total + weight * torch.softmax(model(x), dim=1)
        return total[0].cpu().numpy()
    return probs


def deletion_curve(
    probs_fn: ProbsFn,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    class_idx: int,
    percents=DELETION_PERCENTS,
) -> list[float]:
    """Confidence in `class_idx` as the most important pixels (by |heatmap|) are blanked.

    Blanking sets pixels to 0 in normalised space, i.e. the ImageNet mean colour. A faithful
    explanation makes the curve fall quickly. Works for any heatmap (Grad-CAM, SHAP, ...), so
    methods are compared under identical conditions.
    """
    height, width = heatmap.shape
    order = np.argsort(-np.abs(heatmap).ravel(), kind="stable")
    curve = []
    for pct in percents:
        k = int(order.size * pct / 100)
        modified = image_tensor.clone()
        if k:
            rows, cols = np.divmod(order[:k], width)
            modified[:, :, rows, cols] = 0.0
        curve.append(float(probs_fn(modified)[class_idx]))
    return curve


def aopc(curve: list[float]) -> float:
    """Mean drop from the undeleted confidence. Higher = the heatmap found pixels that matter."""
    return float(np.mean([curve[0] - c for c in curve[1:]]))
