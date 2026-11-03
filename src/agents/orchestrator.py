from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import anthropic
import torch
from PIL import Image

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import (
    ClinicalSummary,
    Predictions,
    ScreeningStatus,
    Severity,
    XAISummary,
)
from src.agents.clinical_summarizer import summarize
from src.agents.severity_triage import triage
from src.observability.redaction import span_attributes
from src.observability.tracing import (
    SPAN_ENSEMBLE,
    SPAN_EXPLAINER,
    SPAN_QUALITY_GATE,
    SPAN_SCREENING,
    SPAN_SUMMARIZER,
    SPAN_TRIAGE,
    set_attrs,
    span,
)
from src.preprocessing.inference_transform import preprocess_image
from src.preprocessing.pipeline import preprocess
from src.utils.config import PreprocessingConfig

logger = logging.getLogger(__name__)

RECAPTURE_MESSAGE = (
    "We could not get a reliable result from this photo. Please retake it in good, even "
    "lighting, in focus, with the lesion filling most of the frame."
)


class Predictor(Protocol):
    """What the orchestrator needs from the ensemble (SkinEnsemble satisfies this)."""

    def predict(self, image_tensor: torch.Tensor) -> Predictions: ...


# (model-input tensor, predictions) -> XAISummary. Injected so the Grad-CAM implementation
# can live outside this module; there is deliberately no default.
Explainer = Callable[[torch.Tensor, Predictions], XAISummary]


@dataclass(frozen=True)
class ScreeningResult:
    status: ScreeningStatus
    summary: ClinicalSummary | None = None       # None unless status is COMPLETED
    predictions: Predictions | None = None       # None when REJECTED
    xai: XAISummary | None = None
    messages: list[str] = field(default_factory=list)       # patient-readable: why rejected/blocked
    warnings: list[str] = field(default_factory=list)       # e.g. borderline blur
    triage_reasons: list[str] = field(default_factory=list)  # which safety rules fired (audit)
    escalated: bool = False
    # Contract for the (future) treatment advisor: False for urgent, referral and re-capture
    # cases, so self-treatment advice is never produced for them.
    treatment_advice_allowed: bool = False


async def run_screening(
    image: Image.Image,
    patient: PatientMetadata,
    *,
    ensemble: Predictor,
    explainer: Explainer,
    client: anthropic.AsyncAnthropic | None = None,
    preprocess_config: PreprocessingConfig | None = None,
    tracer: Any = None,
) -> ScreeningResult:
    """Image + metadata -> triaged clinical summary.

    Order: preprocess/quality gate -> ensemble -> explainer -> LLM summarizer -> triage.
    The LLM proposes; `triage` (plain Python) has the last word on severity and blocking.
    Model and explainer calls are blocking, so they run in a worker thread.

    `tracer` is optional and purely observational: with it None (the default) every
    `span()` here yields None and does nothing, so behaviour is byte-identical to an
    uninstrumented run. Spans wrap *decisions*, not functions -- `triage`'s rules_fired is
    the safety audit trail and is the reason this instrumentation exists at all.
    """
    with span(tracer, SPAN_SCREENING, span_attributes(patient)) as root:
        with span(tracer, SPAN_QUALITY_GATE) as gate:
            pre = preprocess(image, preprocess_config)
            # "warn" is a pass that carried warnings (borderline blur): a distinct outcome
            # from a clean pass, and the one worth watching for drift in capture quality.
            if not pre.quality.ok:
                decision = "reject"
            elif pre.quality.warnings:
                decision = "warn"
            else:
                decision = "pass"
            # Counts only: quality.reasons are patient-readable strings and stay off spans.
            set_attrs(gate, {
                "quality.ok": pre.quality.ok,
                "quality.decision": decision,
                "quality.reason_count": len(pre.quality.reasons),
                "quality.warning_count": len(pre.quality.warnings),
            })

        if not pre.quality.ok:
            logger.info("screening rejected by quality gate: %d reason(s)", len(pre.quality.reasons))
            set_attrs(root, {
                "screening.status": ScreeningStatus.REJECTED,
                "screening.escalated": False,
                "screening.treatment_advice_allowed": False,
            })
            return ScreeningResult(
                status=ScreeningStatus.REJECTED,
                messages=list(pre.quality.reasons),
                warnings=list(pre.quality.warnings),
            )

        tensor = preprocess_image(pre.image)
        with span(tracer, SPAN_ENSEMBLE) as sp_ensemble:
            predictions = await asyncio.to_thread(ensemble.predict, tensor)
            set_attrs(sp_ensemble, {
                "ensemble.primary_label": predictions.primary_label,
                "ensemble.primary_confidence": predictions.primary_confidence,
                "ensemble.secondary_label": predictions.secondary_label,
                "ensemble.secondary_confidence": predictions.secondary_confidence,
                "ensemble.agreement_score": predictions.agreement_score,
                # From the schema method, never re-derived: one definition of the thresholds.
                "ensemble.confidence_tier": predictions.confidence_tier(),
            })

        with span(tracer, SPAN_EXPLAINER) as sp_xai:
            xai = await asyncio.to_thread(explainer, tensor, predictions)
            set_attrs(sp_xai, {
                "xai.focus_description": xai.focus_description,
                "xai.boundary_alignment": xai.boundary_alignment,
                "xai.artifact_flag": xai.artifact_flag,
            })

        with span(tracer, SPAN_SUMMARIZER) as sp_sum:
            proposed = await summarize(predictions, patient, xai, client=client)
            # Decisions only. The summary text, recommended_action, caveats and the prompt
            # context are patient-derived content and never reach a span.
            set_attrs(sp_sum, {
                "summarizer.severity_suggested": proposed.severity,
                "summarizer.confidence_level": proposed.confidence_level,
                "summarizer.recapture_needed": proposed.recapture_needed,
                "summarizer.immediate_referral": proposed.immediate_referral,
            })

        with span(tracer, SPAN_TRIAGE) as sp_triage:
            result = triage(predictions, proposed, patient)
            # The safety audit: before/after together are what prove the invariant that
            # rules only escalate or block, never downgrade.
            set_attrs(sp_triage, {
                "triage.rules_fired": result.reasons,
                "triage.severity_before": proposed.severity,
                "triage.severity_after": result.summary.severity,
                "triage.escalated": result.escalated,
                "triage.blocked": result.blocked,
            })

        logger.info(
            "screening done: primary=%s blocked=%s escalated=%s rules=%s",
            predictions.primary_label.value, result.blocked, result.escalated, result.reasons,
        )

        if result.blocked:
            # No `screening.final_severity`: a blocked result withholds the summary from
            # the caller, so publishing a severity for it would describe a verdict that was
            # never shown. `triage.severity_after` is on the triage span for auditors.
            set_attrs(root, {
                "screening.status": ScreeningStatus.BLOCKED,
                "screening.escalated": result.escalated,
                "screening.treatment_advice_allowed": False,
            })
            return ScreeningResult(
                status=ScreeningStatus.BLOCKED,
                predictions=predictions,
                xai=xai,
                messages=[RECAPTURE_MESSAGE],
                warnings=list(pre.quality.warnings),
                triage_reasons=result.reasons,
            )

        final = result.summary
        advice_allowed = (
            final.severity is not Severity.URGENT
            and not final.immediate_referral
            and not final.recapture_needed
        )
        set_attrs(root, {
            "screening.status": ScreeningStatus.COMPLETED,
            "screening.escalated": result.escalated,
            "screening.treatment_advice_allowed": advice_allowed,
            "screening.final_severity": final.severity,
        })
        return ScreeningResult(
            status=ScreeningStatus.COMPLETED,
            summary=final,
            predictions=predictions,
            xai=xai,
            warnings=list(pre.quality.warnings),
            triage_reasons=result.reasons,
            escalated=result.escalated,
            treatment_advice_allowed=advice_allowed,
        )
