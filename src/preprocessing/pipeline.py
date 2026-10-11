from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageOps

from src.preprocessing.artifact_removal import remove_artifacts
from src.preprocessing.color_normalization import shades_of_gray
from src.preprocessing.hair_removal import remove_hair
from src.preprocessing.quality_check import QualityResult, check_quality
from src.utils.config import PreprocessingConfig, get_preprocessing_config


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image                                   # RGB, upright; cleaned only if a step ran
    quality: QualityResult                               # quality.ok False -> do not predict
    steps_applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)       # per-step detail, for logs


def preprocess(image: Image.Image, config: PreprocessingConfig | None = None) -> PreprocessResult:
    """Quality gate first, then the optional pixel-changing steps (all off by default).

    Order: orient + RGB -> quality check -> hair -> artifacts -> colour normalisation.
    Artifact removal runs before colour normalisation because ink is detected by colour.
    The model-input resize and ImageNet normalisation stay in `inference_transform`.
    """
    cfg = config or get_preprocessing_config()

    # Phone photos carry their orientation in EXIF; without this they can reach the model sideways.
    image = ImageOps.exif_transpose(image).convert("RGB")

    quality = check_quality(image, cfg.quality)
    if not quality.ok:
        return PreprocessResult(image, quality)

    steps: list[str] = []
    notes: list[str] = []
    plan = (
        ("remove_hair", cfg.steps.remove_hair, remove_hair),
        ("remove_artifacts", cfg.steps.remove_artifacts, remove_artifacts),
        ("normalize_color", cfg.steps.normalize_color, shades_of_gray),
    )
    for name, enabled, step in plan:
        if not enabled:
            continue
        result = step(image)
        notes.append(f"{name}: {result.detail}")
        if result.applied:
            image = result.image
            steps.append(name)

    return PreprocessResult(image, quality, steps, notes)
