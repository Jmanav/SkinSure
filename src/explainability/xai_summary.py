from __future__ import annotations

import cv2
import numpy as np

from api.schemas.predictions import BoundaryAlignment, FocusRegion, XAISummary
from src.preprocessing.artifact_removal import MAX_MASK_FRACTION, artifact_mask

# HEURISTICS - NOT VALIDATED. Every threshold below is a first guess. The lesion outline is a
# crude brightness segmentation, not a trained segmenter. Tune on a held-out set with
# annotated lesion masks before trusting `boundary_alignment` or `focus_description`.
HOT_THRESHOLD = 0.5            # heatmap value above which a pixel counts as "attended"
DIFFUSE_HOT_FRACTION = 0.5     # attended area above this share of the image -> diffuse
ARTIFACT_FLAG_OVERLAP = 0.25   # share of attended pixels on an artifact -> artifact_flag
ARTIFACT_FOCUS_OVERLAP = 0.5   # share above which the focus itself is the artifact
BACKGROUND_INSIDE_FRACTION = 0.3   # share of attended pixels inside the lesion below this -> background
BORDER_RING_FRACTION = 0.5     # share of in-lesion attention that sits on the rim -> border
IOU_GOOD = 0.5
IOU_PARTIAL = 0.25
LESION_MIN_AREA = 0.03         # lesion mask outside [3%, 70%] of the image is treated as unreliable
LESION_MAX_AREA = 0.70
RIM_WIDTH_FRACTION = 0.25      # rim thickness as a share of the lesion's equivalent radius


def estimate_lesion_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Rough lesion outline from lightness (Otsu on a blurred L channel).

    Assumes the lesion does not touch the image border, so the side of the threshold that
    touches the border least is taken as the lesion. Returns a boolean mask, or None when no
    plausible lesion is found (flat image, mask too small or too large).
    """
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    lightness = cv2.GaussianBlur(lightness, (0, 0), sigmaX=3)
    _, dark = cv2.threshold(lightness, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def border_share(mask: np.ndarray) -> float:
        frame = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        return float(frame.mean())

    candidate = dark if border_share(dark) <= border_share(1 - dark) else 1 - dark
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    area = float(mask.mean())
    if not LESION_MIN_AREA <= area <= LESION_MAX_AREA:
        return None
    return mask.astype(bool)


def summarize_heatmap(heatmap: np.ndarray, rgb: np.ndarray) -> XAISummary:
    """Reduce a [0, 1] heatmap to the three facts the clinical summarizer consumes.

    `heatmap` and `rgb` must be the same height and width (the 224 px model input).
    When the evidence is not usable the fields say "unknown" instead of guessing.
    """
    if heatmap.shape != rgb.shape[:2]:
        raise ValueError(f"heatmap {heatmap.shape} and image {rgb.shape[:2]} sizes differ")

    hot = heatmap >= HOT_THRESHOLD
    if heatmap.max() <= 0 or not hot.any():
        return XAISummary(
            focus_description=FocusRegion.UNKNOWN,
            boundary_alignment=BoundaryAlignment.UNKNOWN,
            artifact_flag=False,
        )
    hot_count = int(hot.sum())

    # Artifacts: same detectors as the preprocessing step. An oversized mask means the
    # detector is misfiring (as in `remove_artifacts`), so it is ignored.
    artifacts = artifact_mask(rgb) > 0
    if artifacts.mean() > MAX_MASK_FRACTION:
        artifacts = np.zeros_like(artifacts)
    artifact_overlap = float((hot & artifacts).sum()) / hot_count
    artifact_flag = artifact_overlap >= ARTIFACT_FLAG_OVERLAP

    lesion = estimate_lesion_mask(rgb)
    if lesion is None:
        alignment = BoundaryAlignment.UNKNOWN
    else:
        union = int((hot | lesion).sum())
        iou = float((hot & lesion).sum()) / union if union else 0.0
        if iou >= IOU_GOOD:
            alignment = BoundaryAlignment.GOOD
        elif iou >= IOU_PARTIAL:
            alignment = BoundaryAlignment.PARTIAL
        else:
            alignment = BoundaryAlignment.POOR

    if artifact_overlap >= ARTIFACT_FOCUS_OVERLAP:
        focus = FocusRegion.ARTIFACT
    elif hot.mean() > DIFFUSE_HOT_FRACTION:
        focus = FocusRegion.DIFFUSE
    elif lesion is None:
        focus = FocusRegion.UNKNOWN
    else:
        inside = hot & lesion
        if inside.sum() / hot_count < BACKGROUND_INSIDE_FRACTION:
            focus = FocusRegion.BACKGROUND_SKIN
        else:
            radius = np.sqrt(lesion.sum() / np.pi)
            width = max(3, int(RIM_WIDTH_FRACTION * radius))
            # elliptical kernel: a square one would erode corners harder and fatten the rim
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
            core = cv2.erode(lesion.astype(np.uint8), kernel)
            rim = lesion & ~core.astype(bool)
            on_rim = (hot & rim).sum() / max(int(inside.sum()), 1)
            focus = (
                FocusRegion.LESION_BORDER if on_rim >= BORDER_RING_FRACTION
                else FocusRegion.LESION_CENTRE
            )

    return XAISummary(
        focus_description=focus, boundary_alignment=alignment, artifact_flag=artifact_flag
    )
