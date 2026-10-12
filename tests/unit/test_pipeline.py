import io

import numpy as np
import pytest
from helpers import flat_image, hair_image, noise_image
from PIL import Image

from src.preprocessing import pipeline
from src.preprocessing.pipeline import preprocess
from src.preprocessing.step_result import StepResult
from src.utils.config import (
    PreprocessingConfig,
    QualityConfig,
    StepsConfig,
    get_preprocessing_config,
)

QUALITY = QualityConfig(
    min_short_side=200, blur_reject_below=5.0, blur_warn_below=15.0, blur_measure_max_side=512
)


def _cfg(**steps) -> PreprocessingConfig:
    return PreprocessingConfig(quality=QUALITY, steps=StepsConfig(**steps))


def _px(image: Image.Image) -> tuple:
    return image.getpixel((0, 0))


# --- shipped defaults --------------------------------------------------------------------------

def test_shipped_config_has_every_pixel_changing_step_off():
    # The models were trained with only resize + ImageNet normalisation.
    steps = get_preprocessing_config().steps
    assert (steps.remove_hair, steps.remove_artifacts, steps.normalize_color) == (False, False, False)


def test_defaults_leave_pixels_unchanged():
    img = noise_image(256)
    result = preprocess(img)

    assert result.quality.ok is True
    assert result.steps_applied == []
    assert np.array_equal(np.asarray(result.image), np.asarray(img))


# --- gate ---------------------------------------------------------------------------------------

def test_rejected_image_runs_no_steps(monkeypatch):
    def boom(image):
        raise AssertionError("step must not run for a rejected image")

    monkeypatch.setattr(pipeline, "remove_hair", boom)
    monkeypatch.setattr(pipeline, "remove_artifacts", boom)
    monkeypatch.setattr(pipeline, "shades_of_gray", boom)

    cfg = _cfg(remove_hair=True, remove_artifacts=True, normalize_color=True)
    result = preprocess(noise_image(64), cfg)

    assert result.quality.ok is False
    assert result.quality.reasons
    assert result.steps_applied == []
    assert result.image.size == (64, 64)


# --- normalisation of the input ----------------------------------------------------------------

def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6                              # rotate 90 deg clockwise to display upright
    buf = io.BytesIO()
    noise_image((300, 200)).save(buf, format="PNG", exif=exif)
    buf.seek(0)

    result = preprocess(Image.open(buf), _cfg())

    assert result.quality.ok is True
    assert result.image.size == (200, 300)


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_other_modes_come_out_rgb(mode):
    img = noise_image(256).convert(mode)
    assert preprocess(img, _cfg()).image.mode == "RGB"


# --- steps -------------------------------------------------------------------------------------

def test_enabled_step_runs_and_is_recorded():
    result = preprocess(hair_image(), _cfg(remove_hair=True))

    assert result.quality.ok is True
    assert result.steps_applied == ["remove_hair"]
    assert result.notes[0].startswith("remove_hair:")


def test_step_order_and_bookkeeping(monkeypatch):
    seen = []

    def hair(image):
        seen.append(("hair", _px(image)))
        return StepResult(flat_image(256, (1, 1, 1)), True, "h")

    def artifacts(image):                         # declines: its image must be discarded
        seen.append(("artifacts", _px(image)))
        return StepResult(flat_image(256, (2, 2, 2)), False, "none")

    def colour(image):
        seen.append(("colour", _px(image)))
        return StepResult(flat_image(256, (3, 3, 3)), True, "c")

    monkeypatch.setattr(pipeline, "remove_hair", hair)
    monkeypatch.setattr(pipeline, "remove_artifacts", artifacts)
    monkeypatch.setattr(pipeline, "shades_of_gray", colour)

    cfg = _cfg(remove_hair=True, remove_artifacts=True, normalize_color=True)
    result = preprocess(noise_image(256), cfg)

    assert [name for name, _ in seen] == ["hair", "artifacts", "colour"]
    assert seen[1][1] == (1, 1, 1)                # artifacts received hair's output
    assert seen[2][1] == (1, 1, 1)                # colour got hair's output, not the declined one
    assert result.steps_applied == ["remove_hair", "normalize_color"]
    assert [n.split(":")[0] for n in result.notes] == [
        "remove_hair", "remove_artifacts", "normalize_color",
    ]
    assert _px(result.image) == (3, 3, 3)


def test_disabled_steps_are_not_called(monkeypatch):
    def boom(image):
        raise AssertionError("disabled step was called")

    monkeypatch.setattr(pipeline, "remove_artifacts", boom)
    monkeypatch.setattr(pipeline, "shades_of_gray", boom)

    result = preprocess(hair_image(), _cfg(remove_hair=True))
    assert result.steps_applied == ["remove_hair"]
