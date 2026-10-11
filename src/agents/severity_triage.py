from __future__ import annotations

from dataclasses import dataclass, field

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import (
    ClinicalSummary,
    ConfidenceLevel,
    DiseaseClass,
    Predictions,
    Severity,
)

# "LLM interprets; Python decides": these rules run AFTER the summarizer and can only make
# the outcome safer (escalate or block) - never downgrade what the LLM proposed.

MELANOMA_LABEL = DiseaseClass.MELANOMA
MELANOMA_URGENT_MIN_CONFIDENCE = 0.40  # at or above this, a melanoma top-1 is always urgent
BLOCK_BELOW_CONFIDENCE = 0.40          # below this the prediction is unreliable: re-capture
INFECTION_LABELS = {DiseaseClass.TINEA}  # classes treated as infections for the spreading rule


@dataclass(frozen=True)
class TriageResult:
    summary: ClinicalSummary            # final summary after overrides
    blocked: bool = False               # True -> do not show a result, ask for a new photo
    escalated: bool = False             # True -> a rule raised severity above the LLM's proposal
    reasons: list[str] = field(default_factory=list)  # which rules fired (for logs/audit)


def triage(
    predictions: Predictions, summary: ClinicalSummary, patient: PatientMetadata
) -> TriageResult:
    """Apply the deterministic safety overrides to the LLM-proposed summary."""
    reasons: list[str] = []
    updates: dict = {}
    caveats = list(summary.confidence_caveats)

    # Rule 1: low model confidence -> block and request re-capture.
    blocked = predictions.primary_confidence < BLOCK_BELOW_CONFIDENCE
    if blocked:
        reasons.append(
            f"primary_confidence {predictions.primary_confidence:.2f} < {BLOCK_BELOW_CONFIDENCE:.2f}"
        )
        caveats.append("Model confidence is too low for a reliable result; please retake the photo.")
        updates.update(recapture_needed=True, confidence_level=ConfidenceLevel.LOW)

    # Rule 2: melanoma at moderate+ confidence -> urgent + immediate referral.
    # Not applied when blocked: a result that is being withheld must not be presented as urgent.
    escalated = False
    is_melanoma = predictions.primary_label == MELANOMA_LABEL
    if (
        not blocked
        and is_melanoma
        and predictions.primary_confidence >= MELANOMA_URGENT_MIN_CONFIDENCE
    ):
        reasons.append(f"melanoma at confidence {predictions.primary_confidence:.2f}")
        escalated = summary.severity != Severity.URGENT
        updates.update(severity=Severity.URGENT, immediate_referral=True)

    # Rule 3: rapidly spreading infection -> urgent + immediate referral (also skipped when blocked).
    if (
        not blocked
        and patient.is_spreading_rapidly
        and predictions.primary_label in INFECTION_LABELS
    ):
        reasons.append(f"rapidly spreading infection ({predictions.primary_label.value})")
        escalated = escalated or summary.severity != Severity.URGENT
        updates.update(severity=Severity.URGENT, immediate_referral=True)

    if caveats != summary.confidence_caveats:
        updates["confidence_caveats"] = caveats

    final = summary.model_copy(update=updates) if updates else summary
    return TriageResult(summary=final, blocked=blocked, escalated=escalated, reasons=reasons)
