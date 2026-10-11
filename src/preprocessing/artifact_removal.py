from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.preprocessing.step_result import StepResult

MAX_MASK_FRACTION = 0.10  # more than this -> detector is misfiring, leave the image alone


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Saturated blue/green/purple pen marks. Skin hues sit at low OpenCV hue values (red-orange),
    so hues 35-160 with high saturation are not skin."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h >= 35) & (h <= 160)) & (s > 100) & (v > 50)).astype(np.uint8) * 255


def _glare_mask(rgb: np.ndarray) -> np.ndarray:
    """Small, bright, unsaturated blobs: air bubbles and specular glare. Large bright regions
    (e.g. a white background) are ignored."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bright = ((hsv[..., 2] >= 245) & (hsv[..., 1] <= 40)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    max_area = 0.005 * bright.size
    mask = np.zeros_like(bright)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            mask[labels == i] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _ruler_mask(rgb: np.ndarray) -> np.ndarray:
    """Long straight lines (ruler edges and tick rows). Lesion borders are curved and short,
    so only lines at least half the short side long are taken."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_len = int(0.5 * min(gray.shape))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_len, maxLineGap=5)
    mask = np.zeros(gray.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=5)
    return mask


def remove_artifacts(image: Image.Image, inpaint_radius: int = 3) -> StepResult:
    """Remove pen ink, bubbles/glare and ruler lines by inpainting.

    Heuristic detectors, not learned ones: they can miss artifacts and, rarely, flag a real
    feature. If the combined mask covers more than MAX_MASK_FRACTION of the image the step
    is skipped. Runs before colour normalisation because the ink detector relies on colour.
    """
    rgb = np.asarray(image.convert("RGB"))
    mask = _ink_mask(rgb) | _glare_mask(rgb) | _ruler_mask(rgb)

    fraction = float((mask > 0).mean())
    if fraction == 0:
        return StepResult(image, False, "no artifacts detected")
    if fraction > MAX_MASK_FRACTION:
        return StepResult(image, False, f"skipped: mask covers {fraction:.1%} of the image")

    out = cv2.inpaint(rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return StepResult(Image.fromarray(out), True, f"inpainted {fraction:.1%} of pixels")
