import asyncio

import numpy as np
import pytest
from helpers import FakeClient, llm_json
from PIL import Image

from api.schemas.predictions import DiseaseClass, ScreeningStatus, Severity
from src.agents.orchestrator import RECAPTURE_MESSAGE, run_screening


def _noise_image(size: int = 256) -> Image.Image:
    """Random noise: big enough and sharp enough to pass the quality gate."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


class FakeEnsemble:
    def __init__(self, predictions):
        self._predictions = predictions
        self.calls = 0

    def predict(self, image_tensor):
        self.calls += 1
        assert tuple(image_tensor.shape) == (1, 3, 224, 224)
        return self._predictions


class FakeExplainer:
    def __init__(self, xai):
        self._xai = xai
        self.calls = 0

    def __call__(self, image_tensor, predictions):
        self.calls += 1
        return self._xai


def _screen(image, patient, predictions, xai, llm_text=None):
    ensemble, explainer = FakeEnsemble(predictions), FakeExplainer(xai)
    client = FakeClient(llm_text or llm_json())
    result = asyncio.run(
        run_screening(image, patient, ensemble=ensemble, explainer=explainer, client=client)
    )
    return result, ensemble, explainer, client


def test_bad_image_is_rejected_before_the_model_runs(make_predictions, make_patient, make_xai):
    tiny = _noise_image(size=64)
    result, ensemble, explainer, client = _screen(
        tiny, make_patient(), make_predictions(), make_xai()
    )

    assert result.status is ScreeningStatus.REJECTED
    assert result.summary is None and result.predictions is None
    assert result.messages and "too small" in result.messages[0]
    assert (ensemble.calls, explainer.calls, len(client.calls)) == (0, 0, 0)


def test_happy_path_returns_summary(make_predictions, make_patient, make_xai):
    result, ensemble, explainer, client = _screen(
        _noise_image(), make_patient(), make_predictions(), make_xai()
    )

    assert result.status is ScreeningStatus.COMPLETED
    assert result.summary.severity is Severity.ROUTINE
    assert result.predictions.primary_label is DiseaseClass.ECZEMA
    assert result.escalated is False
    assert result.treatment_advice_allowed is True
    assert (ensemble.calls, explainer.calls, len(client.calls)) == (1, 1, 1)


def test_triage_overrides_a_hedging_llm(make_predictions, make_patient, make_xai):
    preds = make_predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.7,
        secondary_label=DiseaseClass.KERATOSIS,
    )
    result, *_ = _screen(
        _noise_image(), make_patient(), preds, make_xai(), llm_json(severity="routine")
    )

    assert result.status is ScreeningStatus.COMPLETED
    assert result.summary.severity is Severity.URGENT
    assert result.summary.immediate_referral is True
    assert result.escalated is True
    assert result.treatment_advice_allowed is False
    assert result.triage_reasons


def test_low_confidence_is_blocked_and_summary_withheld(make_predictions, make_patient, make_xai):
    preds = make_predictions(primary_confidence=0.30, secondary_confidence=0.2)
    result, *_ = _screen(_noise_image(), make_patient(), preds, make_xai())

    assert result.status is ScreeningStatus.BLOCKED
    assert result.summary is None
    assert result.messages == [RECAPTURE_MESSAGE]
    assert result.treatment_advice_allowed is False


@pytest.mark.parametrize(
    "llm_text",
    [llm_json(severity="urgent", immediate_referral=True), llm_json(immediate_referral=True)],
)
def test_no_treatment_advice_for_urgent_or_referral(
    make_predictions, make_patient, make_xai, llm_text
):
    result, *_ = _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), llm_text)

    assert result.status is ScreeningStatus.COMPLETED
    assert result.treatment_advice_allowed is False


def test_artifact_flag_forces_recapture_and_no_advice(make_predictions, make_patient, make_xai):
    result, *_ = _screen(
        _noise_image(), make_patient(), make_predictions(), make_xai(artifact_flag=True)
    )

    assert result.summary.recapture_needed is True
    assert result.treatment_advice_allowed is False


def test_result_never_contains_patient_name(make_predictions, make_patient, make_xai):
    patient = make_patient(patient_name="Asha Verma")
    result, _, _, client = _screen(_noise_image(), patient, make_predictions(), make_xai())

    assert "Asha" not in repr(result)
    assert "Asha" not in str(client.calls)


def test_llm_failure_propagates(make_predictions, make_patient, make_xai):
    with pytest.raises(ValueError):
        _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), "not json")
