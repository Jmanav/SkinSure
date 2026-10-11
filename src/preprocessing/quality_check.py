from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.utils.config import QualityConfig, get_preprocessing_config


@dataclass(frozen=True)
class QualityResult:
    ok: bool                                            # False -> do not send to the model
    reasons: list[str] = field(default_factory=list)    # why it was rejected (patient-readable)
    warnings: list[str] = field(default_factory=list)   # accepted, but worth telling the user
    width: int = 0
    height: int = 0
    blur_score: float = 0.0                             # variance of the Laplacian


def blur_score(image: Image.Image, max_side: int = 512) -> float:
    """Variance of the Laplacian on a grayscale copy (long side capped so scores are
    comparable across image sizes). Low = few edges = blurry or featureless."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    scale = max_side / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_quality(image: Image.Image, config: QualityConfig | None = None) -> QualityResult:
    """Reject images that are too small or nearly featureless; warn on borderline blur.

    Blur sharpness differs by disease in the reference data (BCC and Melanoma images are
    naturally smoother than Eczema or Tinea), so the reject threshold is deliberately low:
    a higher one would turn away malignant-class photos far more often.
    """
    cfg = config or get_preprocessing_config().quality
    width, height = image.size
    score = blur_score(image, cfg.blur_measure_max_side)

    reasons: list[str] = []
    warnings: list[str] = []

    if min(width, height) < cfg.min_short_side:
        reasons.append(
            f"Image is too small ({width}x{height} px); the shorter side must be at least "
            f"{cfg.min_short_side} px."
        )
    if score < cfg.blur_reject_below:
        reasons.append(
            f"Image is too blurry or lacks detail (sharpness {score:.1f} < "
            f"{cfg.blur_reject_below:.1f}); please retake it in good focus and light."
        )
    elif score < cfg.blur_warn_below:
        warnings.append(
            f"Image may be slightly blurry (sharpness {score:.1f}); a sharper photo would "
            "give a more reliable result."
        )

    return QualityResult(
        ok=not reasons, reasons=reasons, warnings=warnings,
        width=width, height=height, blur_score=score,
    )
