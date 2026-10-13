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
