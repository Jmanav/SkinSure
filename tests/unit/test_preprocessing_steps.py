import numpy as np
import pytest
from helpers import DARK_SKIN, SKIN, flat_image, hair_image, noise_image
from PIL import Image

from src.preprocessing.artifact_removal import remove_artifacts
from src.preprocessing.color_normalization import shades_of_gray
from src.preprocessing.hair_removal import remove_hair


def _arr(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(int)


def _skin_canvas(size: int = 256, color=SKIN) -> np.ndarray:
    return np.full((size, size, 3), color, dtype=np.uint8)


# --- hair removal ------------------------------------------------------------------------------

def test_hair_lines_are_inpainted():
    rows = (60, 128, 190)
    result = remove_hair(hair_image(rows=rows))

    assert result.applied is True
    out = _arr(result.image)
    for r in rows:
        assert np.abs(out[r] - np.array(SKIN)).max() <= 10   # line pixels now match the skin


def test_no_hair_leaves_image_untouched():
    img = flat_image(256)
    result = remove_hair(img)

    assert result.applied is False
    assert result.detail == "no hair detected"
    assert result.image is img


def test_dense_dark_structure_is_skipped_not_painted_over():
    # Stripes every 6 px: mask is far above MAX_MASK_FRACTION, as for a dark lesion.
    img = hair_image(rows=range(3, 256, 6))
    result = remove_hair(img)

    assert result.applied is False
    assert result.detail.startswith("skipped")
    assert result.image is img


def test_hair_removal_does_not_mutate_input():
    img = hair_image()
    before = _arr(img).copy()
    remove_hair(img)
    assert np.array_equal(_arr(img), before)


# --- artifact removal --------------------------------------------------------------------------

def test_ink_patch_is_inpainted():
    arr = _skin_canvas()
    arr[100:110, 100:110] = (0, 0, 255)                     # saturated blue pen mark
    result = remove_artifacts(Image.fromarray(arr))

    assert result.applied is True
    patch = _arr(result.image)[100:110, 100:110]
    assert np.abs(patch - np.array(SKIN)).max() <= 15


def test_small_glare_blob_is_inpainted():
    arr = _skin_canvas()
    arr[100:108, 100:108] = 255                             # specular highlight / bubble
    result = remove_artifacts(Image.fromarray(arr))

    assert result.applied is True
    patch = _arr(result.image)[100:108, 100:108]
    assert np.abs(patch - np.array(SKIN)).max() <= 15


def test_large_bright_region_is_not_treated_as_glare():
    arr = _skin_canvas()
    arr[50:90, 50:90] = 255                                 # 2.4% of the image, > 0.5% cap
    result = remove_artifacts(Image.fromarray(arr))

    assert result.applied is False
    assert result.detail == "no artifacts detected"


def test_oversized_mask_is_skipped():
    arr = _skin_canvas()
    arr[0:100, 0:100] = (0, 0, 255)                         # ~15% of the image
    img = Image.fromarray(arr)
    result = remove_artifacts(img)

    assert result.applied is False
    assert result.detail.startswith("skipped")
    assert result.image is img


@pytest.mark.parametrize("tone", [SKIN, DARK_SKIN, (90, 60, 45)])
def test_skin_tones_are_not_flagged_as_ink(tone):
    result = remove_artifacts(flat_image(256, tone))

    assert result.applied is False
    assert result.detail == "no artifacts detected"


# --- colour normalisation ----------------------------------------------------------------------

def test_colour_cast_becomes_neutral_and_brightness_is_kept():
    img = flat_image(64, (200, 100, 100))
    result = shades_of_gray(img)

    out = _arr(result.image)
    channel_means = out.reshape(-1, 3).mean(axis=0)
    assert result.applied is True
    assert channel_means.max() - channel_means.min() <= 1
    assert abs(channel_means.mean() - np.mean((200, 100, 100))) <= 1


def test_neutral_gray_is_unchanged():
    img = flat_image(64, (128, 128, 128))
    out = shades_of_gray(img).image
    assert np.array_equal(_arr(out), _arr(img))


def test_all_black_image_does_not_produce_nan_or_crash():
    out = _arr(shades_of_gray(flat_image(32, (0, 0, 0))).image)
    assert out.max() == 0


def test_output_is_rgb_same_size_with_gains_in_detail():
    img = noise_image((120, 80))
    result = shades_of_gray(img)

    assert result.image.mode == "RGB"
    assert result.image.size == img.size
    assert "channel gains" in result.detail
