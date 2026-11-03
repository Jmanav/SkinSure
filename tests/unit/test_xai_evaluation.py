import numpy as np
import pytest
import torch

from src.explainability.evaluation import DELETION_PERCENTS, aopc, deletion_curve

SIZE = 16


def _probs_fn(image_tensor: torch.Tensor) -> np.ndarray:
    """Toy 'model': confidence in class 0 is the mean brightness of the top-left quadrant."""
    quad = image_tensor[0, :, : SIZE // 2, : SIZE // 2]
    p = float(torch.sigmoid(quad.mean() * 4))
    return np.array([p, 1 - p])


def _image() -> torch.Tensor:
    x = torch.zeros(1, 3, SIZE, SIZE)
    x[:, :, : SIZE // 2, : SIZE // 2] = 1.0      # evidence for class 0 lives top-left
    return x


def _heat(top_left_value: float) -> np.ndarray:
    h = np.zeros((SIZE, SIZE))
    h[: SIZE // 2, : SIZE // 2] = top_left_value
    h[SIZE // 2 :, SIZE // 2 :] = 1.0 - top_left_value
    return h


def test_curve_has_one_point_per_step_and_starts_at_original_confidence():
    curve = deletion_curve(_probs_fn, _image(), _heat(1.0), 0)

    assert len(curve) == len(DELETION_PERCENTS)
    assert curve[0] == pytest.approx(_probs_fn(_image())[0])


def test_faithful_heatmap_scores_higher_than_misleading_one():
    good = aopc(deletion_curve(_probs_fn, _image(), _heat(1.0), 0))   # points at the evidence
    bad = aopc(deletion_curve(_probs_fn, _image(), _heat(0.0), 0))    # points away from it

    assert good > bad
    assert good > 0


def test_input_tensor_is_not_modified():
    x = _image()
    before = x.clone()
    deletion_curve(_probs_fn, x, _heat(1.0), 0)

    assert torch.equal(x, before)


def test_aopc_of_flat_curve_is_zero():
    assert aopc([0.8, 0.8, 0.8, 0.8]) == 0.0


def test_full_deletion_blanks_everything():
    curve = deletion_curve(_probs_fn, _image(), _heat(1.0), 0, percents=(0, 100))
    blank = _probs_fn(torch.zeros(1, 3, SIZE, SIZE))[0]

    assert curve[1] == pytest.approx(blank)


class _FakeEnsemble:
    """Minimal stand-in for SkinEnsemble: batch-capable backbones + weights + device.

    `batched_deletion_curve` only needs `.backbones`, `.weights` and `.device`, so this
    exercises it without loading the real checkpoints.
    """

    device = torch.device("cpu")

    class _Backbone(torch.nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # One logit pair per batch row, driven by the top-left quadrant's brightness
            # so the result depends on which pixels were blanked.
            quad = x[:, :, : SIZE // 2, : SIZE // 2].mean(dim=(1, 2, 3)) * self.scale
            return torch.stack([quad, -quad], dim=1)

    def __init__(self) -> None:
        self._b = {"a": self._Backbone(4.0), "b": self._Backbone(2.0)}

    @property
    def backbones(self):
        return self._b

    @property
    def weights(self):
        return (0.6, 0.4)


def test_batched_deletion_curve_matches_reference_implementation():
    """The experiment script's batched scorer must equal the reference curve exactly.

    `scripts/xai_aopc_experiment.py` stacks all deletion levels into one forward pass for
    speed. That is only legitimate if it reproduces `deletion_curve`, so pin it here as
    well as asserting it at runtime.
    """
    # `scripts/` is not an installed package (pyproject packages only src*/api*), so
    # load the module by path rather than relying on rootdir being on sys.path.
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "xai_aopc_experiment.py"
    spec = importlib.util.spec_from_file_location("xai_aopc_experiment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batched_deletion_curve = module.batched_deletion_curve

    ensemble = _FakeEnsemble()

    def probs_fn(image_tensor: torch.Tensor) -> np.ndarray:
        total = 0.0
        for model, weight in zip(ensemble.backbones.values(), ensemble.weights):
            total = total + weight * torch.softmax(model(image_tensor), dim=1)
        return total[0].detach().numpy()

    for heat in (_heat(1.0), _heat(0.0), np.random.default_rng(0).random((SIZE, SIZE))):
        reference = deletion_curve(probs_fn, _image(), heat, 0)
        batched = batched_deletion_curve(ensemble, _image(), heat, 0)

        assert batched == pytest.approx(reference, abs=1e-6)
