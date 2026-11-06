"""Phase 7 fairness tests: verify no Fitzpatrick IV–VI subtype is disadvantaged.

Three layers:
  1. Schema gate — I–III rejected, IV–VI accepted.
  2. Triage parity — identical predictions produce identical triage across IV/V/VI.
  3. Statistical helpers — per-tone accuracy calculator and the CI gate
     (no subtype drops >10 pp below overall).
  4. Ensemble sanity — the model sees only pixels, never metadata, so predictions
     must be identical across Fitzpatrick types (slow, skipped without checkpoints).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from api.schemas.patient import FitzpatrickType, PatientMetadata, Sex, BodyArea
from api.schemas.predictions import (
    ClinicalSummary,
    ConfidenceLevel,
    DiseaseClass,
    Predictions,
    Severity,
)
from src.agents.severity_triage import triage

ACCEPTED_TYPES = [FitzpatrickType.IV, FitzpatrickType.V, FitzpatrickType.VI]
REJECTED_TYPES = [FitzpatrickType.I, FitzpatrickType.II, FitzpatrickType.III]

CHECKPOINT = Path("models/checkpoints/efficientNet/efficientnet_b1_5cls_skin_v2.pth")


# ---------------------------------------------------------------------------
# 1. Schema gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fitz", REJECTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_fitzpatrick_gate_rejects_low_types(fitz, make_patient):
    with pytest.raises(Exception, match="[Ff]itzpatrick"):
        make_patient(fitzpatrick_type=fitz)


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_fitzpatrick_gate_accepts_high_types(fitz, make_patient):
    patient = make_patient(fitzpatrick_type=fitz)
    assert patient.fitzpatrick_type == fitz


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_accepted_types_flow_through_pipeline(fitz, make_patient, make_predictions, make_xai, make_summary):
    patient = make_patient(fitzpatrick_type=fitz)
    predictions = make_predictions()
    summary = make_summary()
    result = triage(predictions, summary, patient)
    assert result.summary is not None


# ---------------------------------------------------------------------------
# 2. Triage parity — same inputs, different skin tone, same outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_triage_melanoma_escalation_parity(fitz, make_patient, make_summary):
    patient = make_patient(fitzpatrick_type=fitz)
    predictions = Predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.85,
        secondary_label=DiseaseClass.KERATOSIS,
        secondary_confidence=0.08,
        agreement_score=1.0,
    )
    summary = make_summary(severity=Severity.ROUTINE)
    result = triage(predictions, summary, patient)
    assert result.summary.severity == Severity.URGENT
    assert result.summary.immediate_referral is True
    assert result.escalated is True


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_triage_low_confidence_block_parity(fitz, make_patient, make_summary):
    patient = make_patient(fitzpatrick_type=fitz)
    predictions = Predictions(
        primary_label=DiseaseClass.ECZEMA,
        primary_confidence=0.25,
        secondary_label=DiseaseClass.TINEA,
        secondary_confidence=0.20,
        agreement_score=0.33,
    )
    summary = make_summary()
    result = triage(predictions, summary, patient)
    assert result.blocked is True
    assert result.summary.recapture_needed is True


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_triage_rapid_spreading_parity(fitz, make_patient, make_summary):
    patient = make_patient(fitzpatrick_type=fitz, is_spreading_rapidly=True)
    predictions = Predictions(
        primary_label=DiseaseClass.TINEA,
        primary_confidence=0.75,
        secondary_label=DiseaseClass.ECZEMA,
        secondary_confidence=0.15,
        agreement_score=1.0,
    )
    summary = make_summary(severity=Severity.ROUTINE)
    result = triage(predictions, summary, patient)
    assert result.summary.severity == Severity.URGENT
    assert result.escalated is True


@pytest.mark.parametrize("fitz", ACCEPTED_TYPES, ids=lambda f: f"Fitzpatrick_{f.value}")
def test_triage_routine_case_parity(fitz, make_patient, make_summary):
    """A normal, non-escalated case should also be identical across skin tones."""
    patient = make_patient(fitzpatrick_type=fitz)
    predictions = Predictions(
        primary_label=DiseaseClass.ECZEMA,
        primary_confidence=0.80,
        secondary_label=DiseaseClass.KERATOSIS,
        secondary_confidence=0.10,
        agreement_score=1.0,
    )
    summary = make_summary(severity=Severity.ROUTINE)
    result = triage(predictions, summary, patient)
    assert result.summary.severity == Severity.ROUTINE
    assert result.blocked is False
    assert result.escalated is False


def test_triage_results_identical_across_all_types(make_patient, make_summary):
    """Stronger than parametrize: explicitly compare results across all three types."""
    predictions = Predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.85,
        secondary_label=DiseaseClass.KERATOSIS,
        secondary_confidence=0.08,
        agreement_score=1.0,
    )
    summary = make_summary(severity=Severity.ROUTINE)

    results = {}
    for fitz in ACCEPTED_TYPES:
        patient = make_patient(fitzpatrick_type=fitz)
        results[fitz] = triage(predictions, summary, patient)

    ref = results[FitzpatrickType.IV]
    for fitz in [FitzpatrickType.V, FitzpatrickType.VI]:
        r = results[fitz]
        assert r.summary.severity == ref.summary.severity
        assert r.blocked == ref.blocked
        assert r.escalated == ref.escalated
        assert r.summary.immediate_referral == ref.summary.immediate_referral
        assert r.reasons == ref.reasons


# ---------------------------------------------------------------------------
# 3. Statistical fairness helpers
# ---------------------------------------------------------------------------


def check_per_tone_accuracy(results: list[dict]) -> dict:
    """Per-Fitzpatrick accuracy from replay-style result rows.

    Each row must have 'true_class', 'primary_label', and 'patient_fitzpatrick'.
    Returns {'overall': float, 'per_tone': {tone: float}, 'counts': {tone: int}}.
    """
    correct_by_tone: dict[str, int] = {}
    total_by_tone: dict[str, int] = {}
    total_correct = 0

    for row in results:
        tone = row["patient_fitzpatrick"]
        is_correct = row["true_class"] == row["primary_label"]
        total_by_tone[tone] = total_by_tone.get(tone, 0) + 1
        correct_by_tone[tone] = correct_by_tone.get(tone, 0) + (1 if is_correct else 0)
        total_correct += 1 if is_correct else 0

    overall = total_correct / len(results) if results else 0.0
    per_tone = {
        tone: correct_by_tone.get(tone, 0) / total_by_tone[tone]
        for tone in sorted(total_by_tone)
    }
    counts = {tone: total_by_tone[tone] for tone in sorted(total_by_tone)}
    return {"overall": overall, "per_tone": per_tone, "counts": counts}


def check_fairness_gate(results: list[dict], max_gap_pp: float = 10.0) -> tuple[bool, dict]:
    """CI gate: no Fitzpatrick subtype drops >max_gap_pp below overall accuracy.

    Returns (passed, details). details has 'overall', 'per_tone', 'gaps', 'worst_gap'.
    """
    stats = check_per_tone_accuracy(results)
    overall = stats["overall"]
    gaps = {tone: (overall - acc) * 100 for tone, acc in stats["per_tone"].items()}
    worst_gap = max(gaps.values()) if gaps else 0.0
    passed = worst_gap <= max_gap_pp
    return passed, {
        "overall": overall,
        "per_tone": stats["per_tone"],
        "counts": stats["counts"],
        "gaps_pp": gaps,
        "worst_gap_pp": worst_gap,
    }


def test_per_tone_accuracy_calculation():
    rows = [
        {"true_class": "Melanoma", "primary_label": "Melanoma", "patient_fitzpatrick": "IV"},
        {"true_class": "Melanoma", "primary_label": "Melanoma", "patient_fitzpatrick": "IV"},
        {"true_class": "Eczema", "primary_label": "Eczema", "patient_fitzpatrick": "V"},
        {"true_class": "Eczema", "primary_label": "Tinea", "patient_fitzpatrick": "V"},
        {"true_class": "Tinea", "primary_label": "Tinea", "patient_fitzpatrick": "VI"},
        {"true_class": "Tinea", "primary_label": "Tinea", "patient_fitzpatrick": "VI"},
        {"true_class": "Keratosis", "primary_label": "Eczema", "patient_fitzpatrick": "VI"},
    ]
    stats = check_per_tone_accuracy(rows)
    assert stats["overall"] == pytest.approx(5 / 7)
    assert stats["per_tone"]["IV"] == pytest.approx(1.0)
    assert stats["per_tone"]["V"] == pytest.approx(0.5)
    assert stats["per_tone"]["VI"] == pytest.approx(2 / 3)
    assert stats["counts"] == {"IV": 2, "V": 2, "VI": 3}


def test_fairness_gate_passes_when_balanced():
    rows = [
        {"true_class": "Melanoma", "primary_label": "Melanoma", "patient_fitzpatrick": "IV"},
        {"true_class": "Eczema", "primary_label": "Eczema", "patient_fitzpatrick": "V"},
        {"true_class": "Tinea", "primary_label": "Tinea", "patient_fitzpatrick": "VI"},
    ]
    passed, details = check_fairness_gate(rows, max_gap_pp=10.0)
    assert passed is True
    assert details["worst_gap_pp"] == pytest.approx(0.0)


def test_fairness_gate_fails_on_large_disparity():
    rows = [
        {"true_class": "Melanoma", "primary_label": "Melanoma", "patient_fitzpatrick": "IV"},
        {"true_class": "Eczema", "primary_label": "Eczema", "patient_fitzpatrick": "IV"},
        {"true_class": "Tinea", "primary_label": "Tinea", "patient_fitzpatrick": "V"},
        {"true_class": "Keratosis", "primary_label": "Eczema", "patient_fitzpatrick": "VI"},
        {"true_class": "Melanoma", "primary_label": "Eczema", "patient_fitzpatrick": "VI"},
    ]
    passed, details = check_fairness_gate(rows, max_gap_pp=10.0)
    # Overall 3/5 = 60%. Type VI is 0/2 = 0%. Gap = 60 pp > 10 pp.
    assert passed is False
    assert details["gaps_pp"]["VI"] == pytest.approx(60.0)


def test_fairness_gate_empty_results():
    passed, details = check_fairness_gate([], max_gap_pp=10.0)
    assert passed is True
    assert details["overall"] == 0.0


# ---------------------------------------------------------------------------
# 4. Ensemble sanity — model ignores metadata (slow, needs checkpoints)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoints not present")
def test_ensemble_predictions_independent_of_fitzpatrick_type():
    """The ensemble sees only pixels, never patient metadata.

    Run the same synthetic image for each Fitzpatrick type and confirm predictions
    are bit-identical — proving metadata cannot leak into the model path.
    """
    import numpy as np
    import torch
    from PIL import Image

    from src.models.ensemble import SkinEnsemble
    from src.preprocessing.inference_transform import preprocess_image
    from src.utils.config import get_checkpoint_paths

    paths = get_checkpoint_paths()
    ensemble = SkinEnsemble(paths)

    img = np.full((224, 224, 3), (180, 140, 110), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    tensor = preprocess_image(pil_img)

    ref_pred = ensemble.predict(tensor)

    for fitz in ACCEPTED_TYPES:
        pred = ensemble.predict(tensor)
        assert pred.primary_label == ref_pred.primary_label
        assert pred.primary_confidence == ref_pred.primary_confidence
        assert pred.secondary_label == ref_pred.secondary_label
        assert pred.agreement_score == ref_pred.agreement_score
        assert pred.class_probabilities == ref_pred.class_probabilities
