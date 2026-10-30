"""Test doubles and synthetic images shared by unit and integration tests."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic: returns a canned text and records the calls.

    The response carries a `usage` block with the same field names the real SDK's
    `Usage` model uses, so anything reading token counts off a response -- cost
    accounting in particular -- can be tested against this fake rather than the API.
    Pass `usage=` to override the counts; the four cache/non-cache fields are always
    present, because a caller that reads `cache_read_input_tokens` off a real
    response gets 0 rather than an AttributeError when no cache was involved.
    """

    def __init__(self, text: str, usage: dict | None = None):
        self._text = text
        self._usage = {
            "input_tokens": 1200,
            "output_tokens": 180,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            **(usage or {}),
        }
        self.calls: list[dict] = []
        self.responses: list = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            usage=SimpleNamespace(**self._usage),
        )
        self.responses.append(response)
        return response

    @property
    def last_usage(self):
        """The `usage` of the most recent response, as cost accounting would read it."""
        return self.responses[-1].usage

    @property
    def last_usage_template(self):
        """The usage shape this fake will return, without needing a call first."""
        return SimpleNamespace(**self._usage)


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
