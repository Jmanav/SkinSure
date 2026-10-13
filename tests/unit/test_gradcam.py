import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.explainability.gradcam import TARGETS, ensemble_heatmap, grad_cam


class ToyCNN(nn.Module):
    """Class-0 logit = mean brightness of the feature map, so the CAM must follow brightness."""

    def __init__(self, channels_last: bool = False, class0_sign: float = 1.0):
        super().__init__()
        self.channels_last = channels_last
        self.feat = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.feat.weight, 1.0)
        self.head = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[class0_sign], [-class0_sign]]))
        # layout adapter so the hooked layer can output NHWC like Swin
        self.layout = nn.Identity()

    def forward(self, x):
        f = torch.relu(self.feat(x))
        f = self.layout(f.permute(0, 2, 3, 1) if self.channels_last else f)
        pooled = f.mean(dim=(1, 2)) if self.channels_last else f.mean(dim=(2, 3))
        return self.head(pooled)


def _image_with_bright_patch(size=32) -> torch.Tensor:
    x = torch.zeros(1, 3, size, size)
    x[:, :, 2:10, 2:10] = 1.0          # bright patch, top-left
    return x


def _peak_quadrant(cam: np.ndarray) -> str:
    r, c = np.unravel_index(np.argmax(cam), cam.shape)
    h, w = cam.shape
    return ("top" if r < h / 2 else "bottom") + "-" + ("left" if c < w / 2 else "right")


def test_heatmap_follows_the_evidence_and_is_scaled():
    model = ToyCNN().eval()
    x = _image_with_bright_patch()
    cam = grad_cam(model, x, 0, model.layout)

    assert cam.shape == (32, 32)
    assert cam.min() == 0.0 and cam.max() == 1.0
    assert _peak_quadrant(cam) == "top-left"


def test_channels_last_layout_gives_same_answer():
    x = _image_with_bright_patch()
    nchw = ToyCNN().eval()
    nhwc = ToyCNN(channels_last=True).eval()

    a = grad_cam(nchw, x, 0, nchw.layout)
    b = grad_cam(nhwc, x, 0, nhwc.layout, channels_last=True)

    assert np.allclose(a, b, atol=1e-6)


def test_class_with_no_positive_evidence_gives_all_zeros_not_nan():
    model = ToyCNN().eval()
    cam = grad_cam(model, _image_with_bright_patch(), 1, model.layout)  # class 1 is anti-correlated

    assert not np.isnan(cam).any()
    assert cam.max() == 0.0


def test_hook_removed_and_no_parameter_grads_left_behind():
    model = ToyCNN().eval()
    grad_cam(model, _image_with_bright_patch(), 0, model.layout)

    assert len(model.layout._forward_hooks) == 0
    assert all(p.grad is None for p in model.parameters())


def test_hook_removed_even_when_forward_fails():
    model = ToyCNN().eval()
    bad = torch.zeros(1, 5, 32, 32)       # wrong channel count -> forward raises

    try:
        grad_cam(model, bad, 0, model.layout)
    except RuntimeError:
        pass
    assert len(model.layout._forward_hooks) == 0


def test_targets_cover_the_three_backbones():
    assert set(TARGETS) == {"efficientnet", "convnext", "swin"}
    assert TARGETS["swin"].channels_last is True
    assert TARGETS["efficientnet"].channels_last is False


def test_ensemble_heatmap_averages_models_with_weights():
    # Two toy "backbones" exposed through the same API SkinEnsemble provides.
    a, b = ToyCNN().eval(), ToyCNN().eval()
    ens = SimpleNamespace(backbones={"efficientnet": a, "convnext": b}, weights=(0.5, 0.5))
    # Point the toy models at their hookable layers.
    TARGETS_BACKUP = dict(TARGETS)
    try:
        from src.explainability import gradcam as g
        g.TARGETS["efficientnet"] = g.CamTarget(lambda m: m.layout)
        g.TARGETS["convnext"] = g.CamTarget(lambda m: m.layout)
        cam = ensemble_heatmap(ens, _image_with_bright_patch(), 0)
    finally:
        TARGETS.update(TARGETS_BACKUP)

    assert cam.shape == (32, 32)
    assert cam.max() == 1.0
    assert _peak_quadrant(cam) == "top-left"
