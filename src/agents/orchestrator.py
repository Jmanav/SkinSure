from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

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
) -> ScreeningResult:
    """Image + metadata -> triaged clinical summary.

    Order: preprocess/quality gate -> ensemble -> explainer -> LLM summarizer -> triage.
    The LLM proposes; `triage` (plain Python) has the last word on severity and blocking.
    Model and explainer calls are blocking, so they run in a worker thread.
    """
    pre = preprocess(image, preprocess_config)
    if not pre.quality.ok:
        logger.info("screening rejected by quality gate: %d reason(s)", len(pre.quality.reasons))
        return ScreeningResult(
            status=ScreeningStatus.REJECTED,
            messages=list(pre.quality.reasons),
            warnings=list(pre.quality.warnings),
        )

    tensor = preprocess_image(pre.image)
    predictions = await asyncio.to_thread(ensemble.predict, tensor)
    xai = await asyncio.to_thread(explainer, tensor, predictions)

    proposed = await summarize(predictions, patient, xai, client=client)
    result = triage(predictions, proposed, patient)

    logger.info(
        "screening done: primary=%s blocked=%s escalated=%s rules=%s",
        predictions.primary_label.value, result.blocked, result.escalated, result.reasons,
    )

    if result.blocked:
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
