from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.preprocessing.step_result import StepResult

MAX_MASK_FRACTION = 0.10  # more than this "hair" -> the mask is probably the lesion, so skip


def remove_hair(
    image: Image.Image,
    threshold: int = 10,
    inpaint_radius: int = 3,
    kernel_size: int | None = None,
) -> StepResult:
    """DullRazor-style hair removal: black-hat filter finds thin dark structures, which are
    then filled in with inpainting.

    The kernel scales with image size so the same hair thickness is caught at any resolution.
    Safety guard: dark lesions (e.g. melanoma) also respond to the black-hat filter, so if
    the mask covers more than MAX_MASK_FRACTION of the image the step is skipped instead of
    painting over the lesion. Hair is also harder to separate on dark skin, so expect misses.
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if kernel_size is None:
        kernel_size = max(9, int(min(gray.shape) / 30)) | 1  # odd, grows with image size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    fraction = float((mask > 0).mean())
    if fraction == 0:
        return StepResult(image, False, "no hair detected")
    if fraction > MAX_MASK_FRACTION:
        return StepResult(image, False, f"skipped: mask covers {fraction:.1%} of the image")

    out = cv2.inpaint(rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return StepResult(Image.fromarray(out), True, f"inpainted {fraction:.1%} of pixels")
