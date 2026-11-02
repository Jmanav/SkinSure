import pytest

from api.schemas.predictions import ConfidenceLevel, DiseaseClass, Severity
from src.agents.severity_triage import (
    BLOCK_BELOW_CONFIDENCE,
    MELANOMA_URGENT_MIN_CONFIDENCE,
    triage,
)
from src.utils.config import get_settings

# Threshold selected on the validation split (see configs/agents.yaml).
RANKED_MELANOMA_DEFAULT = 0.31

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


# --- Rule F4: melanoma ranked below top-1 (config-gated, OFF by default) ------------------------

def _probs(mel, primary_label, primary_conf):
    """A full 5-class distribution with p(melanoma)=mel and the rest padded out."""
    rest = [c for c in DiseaseClass if c not in (MEL, primary_label)]
    spare = max(0.0, 1.0 - mel - primary_conf)
    d = {MEL.value: mel, primary_label.value: primary_conf}
    for i, c in enumerate(rest):
        d[c.value] = spare if i == 0 else 0.0
    return d


def _ranked_mel_preds(make_predictions, p_mel, primary_conf=0.55, primary_label=BCC):
    """Melanoma present at p_mel but NOT the top-1 - the shape Rule 2 cannot see."""
    return make_predictions(
        primary_label=primary_label,
        primary_confidence=primary_conf,
        secondary_label=MEL,
        secondary_confidence=p_mel,
        class_probabilities=_probs(p_mel, primary_label, primary_conf),
    )


@pytest.fixture
def f4(monkeypatch):
    """Turn Rule F4 on (optionally at a given threshold) for one test."""
    def _set(enabled=True, threshold=RANKED_MELANOMA_DEFAULT):
        cfg = get_settings().agents.triage
        monkeypatch.setattr(cfg, "escalate_on_ranked_melanoma", enabled)
        monkeypatch.setattr(cfg, "ranked_melanoma_min_confidence", threshold)
    return _set


def test_f4_is_off_by_default_in_shipped_config():
    # The whole point of the flag: shipped behaviour is unchanged.
    assert get_settings().agents.triage.escalate_on_ranked_melanoma is False


def test_f4_off_leaves_ranked_melanoma_alone(make_predictions, make_summary, make_patient):
    # Today's behaviour exactly: a melanoma at 0.37 behind a BCC top-1 is not urgent.
    preds = _ranked_mel_preds(make_predictions, 0.37)
    result = triage(preds, make_summary(severity=Severity.MONITOR), make_patient())

    assert result.summary.severity is Severity.MONITOR
    assert result.escalated is False
    assert result.reasons == []


def test_f4_on_escalates_ranked_melanoma_above_threshold(
    make_predictions, make_summary, make_patient, f4
):
    f4()
    preds = _ranked_mel_preds(make_predictions, 0.37)
    result = triage(preds, make_summary(severity=Severity.MONITOR), make_patient())

    assert result.summary.severity is Severity.URGENT
    assert result.summary.immediate_referral is True
    assert result.escalated is True
    assert result.blocked is False
    # A distinct audit reason, so a log shows F4 fired rather than Rule 2.
    assert any("ranked below top-1" in r for r in result.reasons)


def test_f4_at_exactly_the_threshold_escalates(make_predictions, make_summary, make_patient, f4):
    f4(threshold=0.31)
    preds = _ranked_mel_preds(make_predictions, 0.31)
    result = triage(preds, make_summary(), make_patient())
    assert result.summary.severity is Severity.URGENT


def test_f4_below_threshold_does_not_escalate(make_predictions, make_summary, make_patient, f4):
    f4(threshold=0.31)
    # The third real miss: p(mel)=0.064. No usable threshold reaches it.
    preds = _ranked_mel_preds(make_predictions, 0.064, primary_conf=0.70)
    result = triage(preds, make_summary(severity=Severity.ROUTINE), make_patient())

    assert result.summary.severity is Severity.ROUTINE
    assert result.escalated is False


def test_f4_does_not_escalate_when_blocked(make_predictions, make_summary, make_patient, f4):
    f4(threshold=0.31)
    # Top-1 below the block threshold: a withheld result must not be shown as urgent.
    preds = _ranked_mel_preds(make_predictions, 0.35, primary_conf=0.39)
    result = triage(preds, make_summary(), make_patient())

    assert result.blocked is True
    assert result.summary.severity is not Severity.URGENT
    assert result.escalated is False
    assert not any("ranked below top-1" in r for r in result.reasons)


def test_f4_falls_back_gracefully_without_class_probabilities(
    make_predictions, make_summary, make_patient, f4
):
    f4(threshold=0.31)
    # No full distribution and melanoma outside the top-2: nothing to read, so the
    # rule must not fire (and must not raise).
    preds = make_predictions(
        primary_label=BCC,
        primary_confidence=0.55,
        secondary_label=DiseaseClass.KERATOSIS,
        secondary_confidence=0.30,
    )
    assert preds.class_probabilities is None
    result = triage(preds, make_summary(severity=Severity.MONITOR), make_patient())

    assert result.summary.severity is Severity.MONITOR
    assert result.escalated is False


def test_f4_uses_secondary_confidence_when_distribution_absent(
    make_predictions, make_summary, make_patient, f4
):
    f4(threshold=0.31)
    # Melanoma ranked second with no class_probabilities: top-2 already carries p(mel).
    preds = make_predictions(
        primary_label=BCC, primary_confidence=0.55,
        secondary_label=MEL, secondary_confidence=0.37,
    )
    result = triage(preds, make_summary(), make_patient())
    assert result.summary.severity is Severity.URGENT


def test_f4_does_not_double_fire_on_melanoma_top1(make_predictions, make_summary, make_patient, f4):
    f4(threshold=0.31)
    # Melanoma IS the top-1: Rule 2 owns this case, F4 must stay out.
    result = triage(_preds(make_predictions, MEL, 0.62), make_summary(), make_patient())

    assert result.summary.severity is Severity.URGENT
    assert any("melanoma at confidence" in r for r in result.reasons)
    assert not any("ranked below top-1" in r for r in result.reasons)


@pytest.mark.parametrize("severity", [Severity.URGENT])
def test_f4_never_downgrades(make_predictions, make_summary, make_patient, f4, severity):
    f4(threshold=0.31)
    # Below threshold and the LLM already said urgent: the rule must leave it urgent.
    preds = _ranked_mel_preds(make_predictions, 0.10, primary_conf=0.70)
    llm = make_summary(severity=severity, immediate_referral=True)
    result = triage(preds, llm, make_patient())

    assert result.summary.severity is Severity.URGENT
    assert result.escalated is False
