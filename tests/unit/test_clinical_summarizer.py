import asyncio
import json
from types import SimpleNamespace

import anthropic
import pytest
from helpers import FakeClient, llm_json as _llm_json

from api.schemas.predictions import ConfidenceLevel, DiseaseClass, FocusRegion, Severity
from src.agents.clinical_summarizer import _DISAGREEMENT_CAVEAT, build_context, summarize


def _run(client, predictions, patient, xai):
    return asyncio.run(summarize(predictions, patient, xai, client=client))


# --- build_context -----------------------------------------------------------------------------

def test_context_omits_patient_name(make_predictions, make_patient, make_xai):
    ctx = build_context(make_predictions(), make_patient(patient_name="Asha Verma"), make_xai())
    assert "Asha" not in ctx and "Verma" not in ctx


def test_context_uses_enum_values_not_enum_reprs(make_predictions, make_patient, make_xai):
    preds = make_predictions(
        primary_label=DiseaseClass.BASAL_CELL_CARCINOMA, secondary_label=DiseaseClass.TINEA
    )
    ctx = build_context(preds, make_patient(), make_xai(focus_description=FocusRegion.LESION_BORDER))

    assert "Basal_Cell_Carcinoma" in ctx
    assert "lesion_border" in ctx
    assert "DiseaseClass." not in ctx and "FocusRegion." not in ctx


def test_context_handles_missing_optional_fields(make_predictions, make_patient, make_xai):
    ctx = build_context(make_predictions(), make_patient(), make_xai())
    assert "Location: not provided" in ctx
    assert "Season: not provided" in ctx
    assert "none reported" in ctx


# --- summarize: parsing ------------------------------------------------------------------------

def test_valid_response_is_parsed(make_predictions, make_patient, make_xai):
    client = FakeClient(_llm_json(severity="monitor", confidence_level="moderate"))
    result = _run(client, make_predictions(), make_patient(), make_xai())

    assert result.severity is Severity.MONITOR
    assert result.confidence_level is ConfidenceLevel.MODERATE


def test_request_carries_system_prompt_and_context(make_predictions, make_patient, make_xai):
    client = FakeClient(_llm_json())
    _run(client, make_predictions(), make_patient(), make_xai())

    (call,) = client.calls
    assert "JSON" in call["system"]
    assert "MODEL PREDICTION" in call["messages"][0]["content"]
    # FakeClient swallows **kwargs, so only an explicit assertion catches a param the
    # real API would 400 on: sampling params are rejected on Haiku 5.5, and effort is
    # the depth control in their place.
    assert "temperature" not in call
    assert call["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "```json\n" + _llm_json() + "\n```",   # prompt demands JSON only; fences must fail
        _llm_json(severity="critical"),          # invented severity
        _llm_json(severity="urgent", immediate_referral=False),  # violates the safety validator
        json.dumps({"severity": "routine"}),     # missing required keys
    ],
)
def test_bad_response_raises_value_error_with_raw(make_predictions, make_patient, make_xai, raw):
    client = FakeClient(raw)
    with pytest.raises(ValueError) as exc:
        _run(client, make_predictions(), make_patient(), make_xai())
    assert raw in str(exc.value)


def test_empty_content_raises_value_error(make_predictions, make_patient, make_xai):
    # Not a parse edge case to shrug at: max_tokens has to cover thinking plus the JSON
    # body on this model, so a response truncated mid-thought arrives with no usable
    # text. That must fail loudly rather than yield a half-built summary.
    client = FakeClient("")
    with pytest.raises(ValueError, match="Could not parse"):
        _run(client, make_predictions(), make_patient(), make_xai())


def test_response_with_no_text_block_raises_value_error(
    make_predictions, make_patient, make_xai, response_with_blocks
):
    # Thinking is on by default on Haiku 5.5, so a response can legitimately contain
    # thinking blocks and no text block at all. summarize() filters on type == "text",
    # which leaves nothing to parse -- assert that, rather than trusting the filter.
    client = FakeClient(_llm_json())
    client.messages.create = response_with_blocks(
        client, [SimpleNamespace(type="thinking", thinking="weighing the differentials")]
    )

    with pytest.raises(ValueError, match="Could not parse"):
        _run(client, make_predictions(), make_patient(), make_xai())


@pytest.mark.parametrize("error_name", ["APITimeoutError", "RateLimitError"])
def test_api_errors_propagate(make_predictions, make_patient, make_xai, make_api_error, error_name):
    # Deliberate: there is no retry or fallback layer yet, so transport failures must
    # surface to the caller instead of being absorbed. These are not ValueErrors, so
    # _parse's except clause cannot catch them by accident -- this test pins that.
    client = FakeClient(_llm_json())

    async def raise_it(**kwargs):
        raise make_api_error(error_name)

    client.messages.create = raise_it

    with pytest.raises(anthropic.APIError) as exc:
        _run(client, make_predictions(), make_patient(), make_xai())
    assert type(exc.value).__name__ == error_name
    assert not isinstance(exc.value, ValueError)


# --- summarize: deterministic post-processing --------------------------------------------------

@pytest.mark.parametrize(
    "llm_level, expected",
    [("high", ConfidenceLevel.MODERATE), ("moderate", ConfidenceLevel.LOW), ("low", ConfidenceLevel.LOW)],
)
def test_artifact_flag_downgrades_confidence_one_tier(
    make_predictions, make_patient, make_xai, llm_level, expected
):
    client = FakeClient(_llm_json(confidence_level=llm_level))
    result = _run(client, make_predictions(), make_patient(), make_xai(artifact_flag=True))

    assert result.confidence_level is expected
    assert result.recapture_needed is True


def test_no_artifact_leaves_confidence_and_recapture(make_predictions, make_patient, make_xai):
    client = FakeClient(_llm_json(confidence_level="high"))
    result = _run(client, make_predictions(), make_patient(), make_xai(artifact_flag=False))

    assert result.confidence_level is ConfidenceLevel.HIGH
    assert result.recapture_needed is False


def test_low_agreement_appends_disagreement_caveat(make_predictions, make_patient, make_xai):
    client = FakeClient(_llm_json())
    result = _run(client, make_predictions(agreement_score=1 / 3), make_patient(), make_xai())

    assert result.confidence_caveats == [_DISAGREEMENT_CAVEAT]


def test_disagreement_caveat_not_duplicated(make_predictions, make_patient, make_xai):
    llm_caveat = "Model disagreement between the three networks."
    client = FakeClient(_llm_json(confidence_caveats=[llm_caveat]))
    result = _run(client, make_predictions(agreement_score=0.0), make_patient(), make_xai())

    assert result.confidence_caveats == [llm_caveat]


@pytest.mark.parametrize("score", [1.0, 2 / 3])
def test_high_and_moderate_agreement_add_no_caveat(make_predictions, make_patient, make_xai, score):
    client = FakeClient(_llm_json())
    result = _run(client, make_predictions(agreement_score=score), make_patient(), make_xai())

    assert result.confidence_caveats == []


def test_summarizer_does_not_escalate_severity(make_predictions, make_patient, make_xai):
    # severity overrides belong to severity_triage, not the summarizer's post-processing
    client = FakeClient(_llm_json(severity="routine"))
    preds = make_predictions(
        primary_label=DiseaseClass.MELANOMA,
        secondary_label=DiseaseClass.KERATOSIS,
        primary_confidence=0.9,
    )
    result = _run(client, preds, make_patient(), make_xai())

    assert result.severity is Severity.ROUTINE
