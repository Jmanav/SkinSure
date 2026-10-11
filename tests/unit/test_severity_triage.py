import pytest

from api.schemas.predictions import ConfidenceLevel, DiseaseClass, Severity
from src.agents.severity_triage import (
    BLOCK_BELOW_CONFIDENCE,
    MELANOMA_URGENT_MIN_CONFIDENCE,
    triage,
)

MEL, TINEA, ECZEMA, BCC = (
    DiseaseClass.MELANOMA,
    DiseaseClass.TINEA,
    DiseaseClass.ECZEMA,
    DiseaseClass.BASAL_CELL_CARCINOMA,
)


def _preds(make_predictions, label, conf, secondary=DiseaseClass.KERATOSIS):
    return make_predictions(
        primary_label=label,
        primary_confidence=conf,
        secondary_label=secondary,
        secondary_confidence=min(0.1, conf),
    )


# --- Rule 2: melanoma -> urgent ----------------------------------------------------------------

@pytest.mark.parametrize("llm_severity", [Severity.ROUTINE, Severity.MONITOR])
def test_melanoma_escalates_even_if_llm_hedged(
    make_predictions, make_summary, make_patient, llm_severity
):
    result = triage(_preds(make_predictions, MEL, 0.62), make_summary(severity=llm_severity), make_patient())

    assert result.summary.severity is Severity.URGENT
    assert result.summary.immediate_referral is True
    assert result.escalated is True
    assert result.blocked is False
    assert any("melanoma" in r for r in result.reasons)


def test_melanoma_already_urgent_is_not_counted_as_escalation(
    make_predictions, make_summary, make_patient
):
    llm = make_summary(severity=Severity.URGENT, immediate_referral=True)
    result = triage(_preds(make_predictions, MEL, 0.9), llm, make_patient())

    assert result.summary.severity is Severity.URGENT
    assert result.escalated is False
    assert result.reasons  # the rule still fired, for the audit trail


def test_melanoma_exactly_at_threshold_is_urgent(make_predictions, make_summary, make_patient):
    result = triage(
        _preds(make_predictions, MEL, MELANOMA_URGENT_MIN_CONFIDENCE), make_summary(), make_patient()
    )
    assert result.summary.severity is Severity.URGENT
    assert result.blocked is False


# --- Rule 1: low confidence -> block -----------------------------------------------------------

def test_low_confidence_is_blocked_with_recapture(make_predictions, make_summary, make_patient):
    result = triage(_preds(make_predictions, ECZEMA, 0.30), make_summary(), make_patient())

    assert result.blocked is True
    assert result.summary.recapture_needed is True
    assert result.summary.confidence_level is ConfidenceLevel.LOW
    assert any("retake" in c.lower() for c in result.summary.confidence_caveats)


def test_just_below_threshold_is_blocked(make_predictions, make_summary, make_patient):
    result = triage(
        _preds(make_predictions, ECZEMA, BLOCK_BELOW_CONFIDENCE - 0.001), make_summary(), make_patient()
    )
    assert result.blocked is True


def test_blocked_melanoma_is_not_presented_as_urgent(make_predictions, make_summary, make_patient):
    result = triage(_preds(make_predictions, MEL, 0.25), make_summary(), make_patient())

    assert result.blocked is True
    assert result.escalated is False
    assert result.summary.severity is Severity.ROUTINE
    assert result.summary.immediate_referral is False


def test_blocked_spreading_tinea_is_not_escalated(make_predictions, make_summary, make_patient):
    result = triage(
        _preds(make_predictions, TINEA, 0.30),
        make_summary(),
        make_patient(is_spreading_rapidly=True),
    )
    assert result.blocked is True
    assert result.summary.severity is Severity.ROUTINE


# --- Rule 3: rapidly spreading infection -------------------------------------------------------

def test_spreading_tinea_is_urgent(make_predictions, make_summary, make_patient):
    result = triage(
        _preds(make_predictions, TINEA, 0.80),
        make_summary(),
        make_patient(is_spreading_rapidly=True),
    )
    assert result.summary.severity is Severity.URGENT
    assert result.summary.immediate_referral is True
    assert result.escalated is True
    assert any("spreading" in r for r in result.reasons)


def test_non_spreading_tinea_is_untouched(make_predictions, make_summary, make_patient):
    result = triage(_preds(make_predictions, TINEA, 0.80), make_summary(), make_patient())
    assert result.summary.severity is Severity.ROUTINE
    assert result.escalated is False
    assert result.reasons == []


@pytest.mark.parametrize("label", [ECZEMA, BCC])
def test_spreading_flag_ignored_for_non_infection_classes(
    make_predictions, make_summary, make_patient, label
):
    result = triage(
        _preds(make_predictions, label, 0.80),
        make_summary(),
        make_patient(is_spreading_rapidly=True),
    )
    assert result.summary.severity is Severity.ROUTINE
    assert result.escalated is False


# --- Never downgrade / pass-through ------------------------------------------------------------

@pytest.mark.parametrize("severity", [Severity.MONITOR, Severity.URGENT])
def test_rules_never_downgrade(make_predictions, make_summary, make_patient, severity):
    llm = make_summary(severity=severity, immediate_referral=severity is Severity.URGENT)
    result = triage(_preds(make_predictions, ECZEMA, 0.9), llm, make_patient())

    assert result.summary.severity is severity
    assert result.escalated is False


def test_no_rule_fires_returns_summary_unchanged(make_predictions, make_summary, make_patient):
    llm = make_summary()
    result = triage(_preds(make_predictions, ECZEMA, 0.9), llm, make_patient())

    assert result.summary == llm
    assert result.blocked is False
    assert result.escalated is False
    assert result.reasons == []


def test_triage_does_not_mutate_input_summary(make_predictions, make_summary, make_patient):
    llm = make_summary(confidence_caveats=["existing"])
    triage(_preds(make_predictions, MEL, 0.9), llm, make_patient())
    triage(_preds(make_predictions, ECZEMA, 0.1), llm, make_patient())

    assert llm.severity is Severity.ROUTINE
    assert llm.confidence_caveats == ["existing"]
    assert llm.recapture_needed is False


def test_existing_caveats_are_kept_when_blocking(make_predictions, make_summary, make_patient):
    llm = make_summary(confidence_caveats=["existing"])
    result = triage(_preds(make_predictions, ECZEMA, 0.1), llm, make_patient())

    assert result.summary.confidence_caveats[0] == "existing"
    assert len(result.summary.confidence_caveats) == 2
