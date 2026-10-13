from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Forward hooks are registered on the shared model modules. Holding this lock for the whole
# computation (and ignoring forwards from other threads, see `grad_cam`) keeps concurrent
# requests from capturing each other's activations.
_LOCK = threading.Lock()


@dataclass(frozen=True)
class CamTarget:
    """Which layer's output to use as the Grad-CAM feature map, and its memory layout."""
    layer: Callable[[nn.Module], nn.Module]
    channels_last: bool = False   # True when the layer outputs (B, H, W, C) instead of (B, C, H, W)


# The last feature map before global pooling in each backbone (timm 1.0.x, 224px input):
#   efficientnet_b1 : backbone.bn2        -> (B, 1280, 7, 7)
#   convnext_tiny   : backbone.stages[-1] -> (B,  768, 7, 7)
#   swin_tiny       : backbone.layers[-1] -> (B, 7, 7,  768)   (NHWC)
TARGETS: dict[str, CamTarget] = {
    "efficientnet": CamTarget(lambda m: m.backbone.bn2),
    "convnext": CamTarget(lambda m: m.backbone.stages[-1]),
    "swin": CamTarget(lambda m: m.backbone.layers[-1], channels_last=True),
}


def _minmax(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    peak = cam.max()
    return cam / peak if peak > 0 else np.zeros_like(cam)


def grad_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    layer: nn.Module,
    channels_last: bool = False,
) -> np.ndarray:
    """Grad-CAM heatmap for one image and one class, scaled to [0, 1].

    `image_tensor` is (1, 3, H, W), already normalised. The result is (H, W). Uses
    `torch.autograd.grad`, so no `.grad` is left on the model's parameters, and the hook is
    always removed. The model should be in eval mode.
    """
    owner = threading.get_ident()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if threading.get_ident() == owner:       # ignore other threads' forwards (e.g. predict)
            captured["act"] = output

    device = next(model.parameters()).device
    with _LOCK:
        handle = layer.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                x = image_tensor.detach().to(device).clone().requires_grad_(True)
                logits = model(x)
                if "act" not in captured:
                    raise RuntimeError("target layer did not run during the forward pass")
                act = captured["act"]
                grad = torch.autograd.grad(logits[0, class_idx], act)[0]
        finally:
            handle.remove()

    if channels_last:
        act, grad = act.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
    weights = grad.mean(dim=(2, 3), keepdim=True)                 # one weight per channel
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))        # keep only positive evidence
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return _minmax(cam[0, 0].detach().cpu().numpy())


def ensemble_heatmap(ensemble, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Weighted average of the three backbones' Grad-CAM maps (same weights as the ensemble).

    `ensemble` is a `SkinEnsemble` (uses its `backbones` and `weights`). Each map is scaled to
    [0, 1] before averaging so no backbone dominates through raw gradient scale.
    """
    maps = []
    for (name, model), weight in zip(ensemble.backbones.items(), ensemble.weights):
        target = TARGETS[name]
        cam = grad_cam(model, image_tensor, class_idx, target.layer(model), target.channels_last)
        maps.append(weight * cam)
    return _minmax(np.sum(maps, axis=0))
