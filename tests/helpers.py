"""Test doubles and synthetic images shared by unit and integration tests."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic: returns a canned text and records the calls."""

    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def llm_json(**overrides) -> str:
    """A valid summarizer response as the LLM would return it."""
    data = dict(
        severity="routine",
        summary="A common skin condition.",
        recommended_action="Moisturise daily.",
        confidence_level="high",
        confidence_caveats=[],
        recapture_needed=False,
        immediate_referral=False,
    )
    data.update(overrides)
    return json.dumps(data)


# --- synthetic images (preprocessing tests) ---------------------------------------------------

SKIN = (200, 150, 120)       # light-medium skin-like RGB
DARK_SKIN = (120, 80, 60)    # Fitzpatrick V-VI-like RGB


def flat_image(size: int | tuple[int, int] = 256, color=SKIN) -> Image.Image:
    """Uniform colour. `size` is a side length or (width, height)."""
    w, h = (size, size) if isinstance(size, int) else size
    return Image.new("RGB", (w, h), color)


def noise_image(size: int | tuple[int, int] = 256, seed: int = 0) -> Image.Image:
    """Random noise: large Laplacian variance, so it always passes the blur gate."""
    w, h = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def hair_image(size: int = 256, rows=(60, 128, 190), color=SKIN) -> Image.Image:
    """Flat skin colour with 1 px black horizontal lines standing in for hairs."""
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    for r in rows:
        arr[r, :, :] = 0
    return Image.fromarray(arr)
