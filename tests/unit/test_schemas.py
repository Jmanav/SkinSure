import pytest
from pydantic import ValidationError

from api.schemas.patient import FitzpatrickType
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


# --- PatientMetadata ---------------------------------------------------------------------------

@pytest.mark.parametrize("ftype", [FitzpatrickType.I, FitzpatrickType.II, FitzpatrickType.III])
def test_fitzpatrick_i_to_iii_rejected(make_patient, ftype):
    with pytest.raises(ValidationError, match="IV - VI"):
        make_patient(fitzpatrick_type=ftype)


@pytest.mark.parametrize("ftype", [FitzpatrickType.IV, FitzpatrickType.V, FitzpatrickType.VI])
def test_fitzpatrick_iv_to_vi_accepted(make_patient, ftype):
    assert make_patient(fitzpatrick_type=ftype).fitzpatrick_type == ftype


def test_fitzpatrick_outside_enum_rejected(make_patient):
    with pytest.raises(ValidationError):
        make_patient(fitzpatrick_type="VII")


@pytest.mark.parametrize("age", [-1, 121])
def test_age_out_of_range_rejected(make_patient, age):
    with pytest.raises(ValidationError):
        make_patient(patient_age=age)


def test_negative_symptom_duration_rejected(make_patient):
    with pytest.raises(ValidationError):
        make_patient(symptom_duration_days=-1)


# --- Predictions -------------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["melanoma", "nevus", "Melanoma ", ""])
def test_unknown_or_miscased_label_rejected(make_predictions, label):
    with pytest.raises(ValidationError):
        make_predictions(primary_label=label)


def test_exact_label_string_accepted(make_predictions):
    p = make_predictions(primary_label="Melanoma", secondary_label="Tinea")
    assert p.primary_label is DiseaseClass.MELANOMA


def test_disease_classes_match_checkpoint_order():
    # model index order is alphabetical (see src/models/CLAUDE.md)
    values = [c.value for c in DiseaseClass]
    assert values == sorted(values)
    assert len(values) == 5


@pytest.mark.parametrize("field", ["primary_confidence", "secondary_confidence", "agreement_score"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probabilities_bounded(make_predictions, field, value):
    with pytest.raises(ValidationError):
        make_predictions(**{field: value})


def test_secondary_label_must_differ(make_predictions):
    with pytest.raises(ValidationError, match="differ"):
        make_predictions(primary_label=DiseaseClass.TINEA, secondary_label=DiseaseClass.TINEA)


def test_secondary_confidence_cannot_exceed_primary(make_predictions):
    with pytest.raises(ValidationError, match="exceed"):
        make_predictions(primary_confidence=0.4, secondary_confidence=0.5)


@pytest.mark.parametrize(
    "score, tier",
    [(1.0, "high"), (2 / 3, "moderate"), (1 / 3, "low"), (0.0, "low")],
)
def test_confidence_tier(make_predictions, score, tier):
    assert make_predictions(agreement_score=score).confidence_tier() == tier


# --- XAISummary --------------------------------------------------------------------------------

def test_free_text_xai_fields_rejected(make_xai):
    with pytest.raises(ValidationError):
        make_xai(focus_description="centre of the lesion")
    with pytest.raises(ValidationError):
        make_xai(boundary_alignment="kind of aligned")


def test_artifact_focus_requires_flag(make_xai):
    with pytest.raises(ValidationError, match="artifact_flag"):
        make_xai(focus_description=FocusRegion.ARTIFACT, artifact_flag=False)


def test_artifact_focus_with_flag_accepted(make_xai):
    xai = make_xai(focus_description=FocusRegion.ARTIFACT, artifact_flag=True)
    assert xai.artifact_flag is True


def test_flag_without_artifact_focus_is_allowed(make_xai):
    # the reverse is legitimate: a minor artifact can be flagged while focus is on the lesion
    xai = make_xai(focus_description=FocusRegion.LESION_CENTRE, artifact_flag=True)
    assert xai.boundary_alignment is BoundaryAlignment.GOOD


# --- ClinicalSummary ---------------------------------------------------------------------------

def test_urgent_requires_immediate_referral(make_summary):
    with pytest.raises(ValidationError, match="immediate_referral"):
        make_summary(severity=Severity.URGENT, immediate_referral=False)


def test_urgent_with_referral_accepted(make_summary):
    s = make_summary(severity=Severity.URGENT, immediate_referral=True)
    assert s.severity is Severity.URGENT


def test_invented_severity_rejected(make_summary):
    with pytest.raises(ValidationError):
        make_summary(severity="critical")


def test_invented_confidence_level_rejected(make_summary):
    with pytest.raises(ValidationError):
        make_summary(confidence_level="certain")


def test_summary_defaults(make_summary):
    s = make_summary()
    assert s.confidence_caveats == []
    assert s.recapture_needed is False
    assert s.immediate_referral is False
    assert s.confidence_level is ConfidenceLevel.HIGH
    assert isinstance(s, ClinicalSummary)


def test_caveat_lists_not_shared_between_instances(make_summary):
    a, b = make_summary(), make_summary()
    a.confidence_caveats.append("x")
    assert b.confidence_caveats == []
