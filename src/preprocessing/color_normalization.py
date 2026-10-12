from __future__ import annotations

import numpy as np
from PIL import Image

from src.preprocessing.step_result import StepResult


def shades_of_gray(image: Image.Image, power: float = 6.0) -> StepResult:
    """Shades-of-Gray colour constancy (Minkowski p-norm of each channel as the illuminant).

    Rescales each channel so the estimated illuminant becomes neutral while keeping the
    overall brightness. CAUTION for this project: the method assumes the scene averages to
    gray. A close-up of skin does not, so it can flatten genuine skin-tone colour, which
    matters for Fitzpatrick IV-VI. It is off by default; enable only after checking
    predictions with and without it.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float64)
    illuminant = np.power(np.mean(np.power(arr, power), axis=(0, 1)), 1.0 / power)
    illuminant = np.maximum(illuminant, 1e-6)
    gain = illuminant.mean() / illuminant            # per-channel gain, mean brightness kept
    out = np.rint(np.clip(arr * gain, 0, 255)).astype(np.uint8)  # round: astype alone truncates
    return StepResult(
        Image.fromarray(out), True,
        f"channel gains R={gain[0]:.2f} G={gain[1]:.2f} B={gain[2]:.2f}",
    )
