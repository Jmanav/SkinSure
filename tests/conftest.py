from types import SimpleNamespace

import pytest

from api.schemas.patient import BodyArea, FitzpatrickType, PatientMetadata, Sex
from api.schemas.predictions import (
    BoundaryAlignment,
    ClinicalSummary,
    ConfidenceLevel,
    DiseaseClass,
    FocusRegion,
    Predictions,
    Severity,
    XAISummary,
)


@pytest.fixture
def make_patient():
    def _make(**overrides) -> PatientMetadata:
        data = dict(
            patient_id=1,
            patient_name="Test Patient",
            patient_age=45,
            patient_sex=Sex.FEMALE,
            fitzpatrick_type=FitzpatrickType.V,
            body_area=BodyArea.ARM,
            symptom_duration_days=30,
        )
        data.update(overrides)
        return PatientMetadata(**data)

    return _make


@pytest.fixture
def make_predictions():
    def _make(**overrides) -> Predictions:
        data = dict(
            primary_label=DiseaseClass.ECZEMA,
            primary_confidence=0.80,
            secondary_label=DiseaseClass.KERATOSIS,
            secondary_confidence=0.10,
            agreement_score=1.0,
        )
        data.update(overrides)
        return Predictions(**data)

    return _make


@pytest.fixture
def make_xai():
    def _make(**overrides) -> XAISummary:
        data = dict(
            focus_description=FocusRegion.LESION_CENTRE,
            boundary_alignment=BoundaryAlignment.GOOD,
            artifact_flag=False,
        )
        data.update(overrides)
        return XAISummary(**data)

    return _make


@pytest.fixture
def make_summary():
    def _make(**overrides) -> ClinicalSummary:
        data = dict(
            severity=Severity.ROUTINE,
            summary="A common, usually harmless skin condition.",
            recommended_action="Keep the area moisturised.",
            confidence_level=ConfidenceLevel.HIGH,
        )
        data.update(overrides)
        return ClinicalSummary(**data)

    return _make


@pytest.fixture
def make_api_error():
    """Build a real SDK transport error by name.

    These need an httpx request/response to construct, so they are built here rather
    than hand-rolled: a stand-in exception would not prove the real ones propagate.
    """
    import anthropic
    import httpx

    def _make(name: str) -> Exception:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        if name == "APITimeoutError":
            return anthropic.APITimeoutError(request=request)
        if name == "RateLimitError":
            return anthropic.RateLimitError(
                "rate limited", response=httpx.Response(429, request=request), body=None
            )
        raise ValueError(f"unhandled error name: {name!r}")

    return _make


@pytest.fixture
def response_with_blocks():
    """Replace a FakeClient's create() with one returning arbitrary content blocks.

    For response shapes the fake's canned text cannot express -- a thinking-only
    response, say -- while still recording the call and reporting usage.
    """
    def _make(client, blocks):
        async def _create(**kwargs):
            client.calls.append(kwargs)
            response = SimpleNamespace(
                content=blocks, usage=client.last_usage_template
            )
            client.responses.append(response)
            return response

        return _create

    return _make
