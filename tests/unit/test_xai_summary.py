import numpy as np
import pytest

from api.schemas.predictions import BoundaryAlignment, FocusRegion, XAISummary
from src.explainability.xai_summary import estimate_lesion_mask, summarize_heatmap

SIZE = 224
SKIN = (200, 150, 120)
LESION = (90, 60, 45)


def _scene(lesion_radius=40, center=(112, 112)) -> np.ndarray:
    """Skin-coloured canvas with one dark disc."""
    img = np.full((SIZE, SIZE, 3), SKIN, dtype=np.uint8)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    img[(yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= lesion_radius**2] = LESION
    return img


def _blob(center, radius) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    return np.clip(1.0 - dist / radius, 0, 1)


def _ring(center, r_in, r_out) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    return ((dist >= r_in) & (dist <= r_out)).astype(float)


# --- lesion outline ---------------------------------------------------------------------------

def test_lesion_disc_is_found():
    mask = estimate_lesion_mask(_scene(lesion_radius=40))
    assert mask is not None
    area = np.pi * 40**2 / SIZE**2
    assert abs(mask.mean() - area) < 0.03


def test_flat_image_has_no_lesion():
    flat = np.full((SIZE, SIZE, 3), SKIN, dtype=np.uint8)
    assert estimate_lesion_mask(flat) is None


# --- focus and alignment ----------------------------------------------------------------------

# _blob(center, R) is "hot" (>= 0.5) out to R / 2, so R = 80 attends a radius-40 lesion exactly.
def test_heat_matching_the_lesion_is_good_alignment():
    xai = summarize_heatmap(_blob((112, 112), 80), _scene(lesion_radius=40))

    assert xai.focus_description is FocusRegion.LESION_CENTRE
    assert xai.boundary_alignment is BoundaryAlignment.GOOD
    assert xai.artifact_flag is False


def test_heat_covering_part_of_the_lesion_is_partial_alignment():
    xai = summarize_heatmap(_blob((112, 112), 48), _scene(lesion_radius=40))   # hot radius 24

    assert xai.focus_description is FocusRegion.LESION_CENTRE
    assert xai.boundary_alignment is BoundaryAlignment.PARTIAL


def test_small_central_focus_is_still_centre_but_poor_alignment():
    # On the lesion, but covering ~9% of it: overlap with the outline is low by definition.
    xai = summarize_heatmap(_blob((112, 112), 24), _scene(lesion_radius=40))   # hot radius 12

    assert xai.focus_description is FocusRegion.LESION_CENTRE
    assert xai.boundary_alignment is BoundaryAlignment.POOR


def test_heat_on_lesion_rim_is_border_focus():
    xai = summarize_heatmap(_ring((112, 112), 32, 40), _scene(lesion_radius=40))

    assert xai.focus_description is FocusRegion.LESION_BORDER


def test_heat_far_from_lesion_is_background_and_poor():
    xai = summarize_heatmap(_blob((30, 30), 20), _scene())

    assert xai.focus_description is FocusRegion.BACKGROUND_SKIN
    assert xai.boundary_alignment is BoundaryAlignment.POOR


def test_heat_everywhere_is_diffuse():
    xai = summarize_heatmap(np.ones((SIZE, SIZE)), _scene())

    assert xai.focus_description is FocusRegion.DIFFUSE


# --- artifacts --------------------------------------------------------------------------------

def test_heat_on_ink_mark_flags_artifact():
    img = _scene()
    img[20:40, 20:40] = (0, 0, 255)                     # blue pen mark away from the lesion
    xai = summarize_heatmap(_blob((30, 30), 15), img)

    assert xai.focus_description is FocusRegion.ARTIFACT
    assert xai.artifact_flag is True


def test_artifact_present_but_not_attended_is_not_flagged():
    img = _scene()
    img[20:40, 20:40] = (0, 0, 255)
    xai = summarize_heatmap(_blob((112, 112), 80), img)  # heat stays on the lesion

    assert xai.artifact_flag is False
    assert xai.focus_description is FocusRegion.LESION_CENTRE


# --- missing evidence -------------------------------------------------------------------------

def test_zero_heatmap_is_unknown_not_a_guess():
    xai = summarize_heatmap(np.zeros((SIZE, SIZE)), _scene())

    assert xai.focus_description is FocusRegion.UNKNOWN
    assert xai.boundary_alignment is BoundaryAlignment.UNKNOWN
    assert xai.artifact_flag is False


def test_no_lesion_outline_gives_unknown_alignment():
    flat = np.full((SIZE, SIZE, 3), SKIN, dtype=np.uint8)
    xai = summarize_heatmap(_blob((112, 112), 30), flat)

    assert xai.boundary_alignment is BoundaryAlignment.UNKNOWN
    assert xai.focus_description is FocusRegion.UNKNOWN


def test_size_mismatch_raises():
    with pytest.raises(ValueError, match="sizes differ"):
        summarize_heatmap(np.ones((100, 100)), _scene())


def test_output_always_satisfies_schema_validators():
    for heat in (_blob((112, 112), 40), _blob((30, 30), 20), np.ones((SIZE, SIZE))):
        assert isinstance(summarize_heatmap(heat, _scene()), XAISummary)
