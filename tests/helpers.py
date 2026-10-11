"""Test doubles shared by unit and integration tests."""
import json
from types import SimpleNamespace


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
