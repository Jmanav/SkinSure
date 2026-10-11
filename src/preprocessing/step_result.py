from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class StepResult:
    """Output of a pixel-changing preprocessing step."""
    image: Image.Image
    applied: bool        # False -> the step chose to leave the image untouched
    detail: str = ""     # e.g. "masked 2.3% of pixels" or why it was skipped
