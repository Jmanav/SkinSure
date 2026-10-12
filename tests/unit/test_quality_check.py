import numpy as np
import pytest
from helpers import flat_image, noise_image

from src.preprocessing.quality_check import blur_score, check_quality
from src.utils.config import QualityConfig, get_preprocessing_config

CFG = QualityConfig(
    min_short_side=200, blur_reject_below=5.0, blur_warn_below=15.0, blur_measure_max_side=512
)


# --- size gate ---------------------------------------------------------------------------------

def test_exactly_min_short_side_is_accepted():
    result = check_quality(noise_image(200), CFG)
    assert result.ok is True
    assert result.reasons == []


def test_one_pixel_below_min_short_side_is_rejected():
    result = check_quality(noise_image(199), CFG)
    assert result.ok is False
    assert "too small" in result.reasons[0]


def test_short_side_decides_not_long_side():
    result = check_quality(noise_image((400, 100)), CFG)  # 400 wide, 100 tall
    assert result.ok is False
    assert "too small" in result.reasons[0]


def test_dimensions_are_reported():
    result = check_quality(noise_image((320, 240)), CFG)
    assert (result.width, result.height) == (320, 240)


# --- blur gate ---------------------------------------------------------------------------------

def test_featureless_image_is_rejected_with_retake_advice():
    result = check_quality(flat_image(256), CFG)
    assert result.ok is False
    assert "blurry" in result.reasons[0]
    assert "retake" in result.reasons[0]


def test_sharp_image_is_accepted_without_warning():
    result = check_quality(noise_image(256), CFG)
    assert result.ok is True
    assert result.warnings == []


def test_borderline_blur_warns_but_accepts():
    sharp = noise_image(256)
    score = blur_score(sharp)
    cfg = QualityConfig(
        min_short_side=200,
        blur_reject_below=5.0,
        blur_warn_below=score + 1.0,   # puts this image just inside the warn band
        blur_measure_max_side=512,
    )
    result = check_quality(sharp, cfg)

    assert result.ok is True
    assert result.reasons == []
    assert len(result.warnings) == 1
    assert "blurry" in result.warnings[0]


def test_every_failing_check_adds_a_reason():
    result = check_quality(flat_image(64), CFG)   # too small AND featureless
    assert result.ok is False
    assert len(result.reasons) == 2


# --- blur_score --------------------------------------------------------------------------------

def test_blur_score_orders_flat_below_noise():
    assert blur_score(flat_image(256)) == pytest.approx(0.0)
    assert blur_score(noise_image(256)) > 1000


def test_blur_score_caps_large_images():
    score = blur_score(noise_image(2000), max_side=512)
    assert np.isfinite(score) and score > 0


# --- shipped defaults (documented in CLAUDE.md) -------------------------------------------------

def test_shipped_quality_thresholds_match_documentation():
    q = get_preprocessing_config().quality
    assert q.min_short_side == 200
    assert q.blur_reject_below == 5.0
    assert q.blur_warn_below == 15.0
