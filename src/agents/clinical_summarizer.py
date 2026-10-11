from __future__ import annotations

import json

import anthropic

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import (
    ClinicalSummary,
    ConfidenceLevel,
    Predictions,
    XAISummary,
)
from src.utils.config import get_settings

_TIER_DOWN = {
    ConfidenceLevel.HIGH: ConfidenceLevel.MODERATE,
    ConfidenceLevel.MODERATE: ConfidenceLevel.LOW,
    ConfidenceLevel.LOW: ConfidenceLevel.LOW,
}

_DISAGREEMENT_CAVEAT = (
    "Model disagreement: the three models did not fully agree on this result, "
    "so it should be treated with extra caution."
)

_SYSTEM_PROMPT = """\
You are a clinical decision-support assistant for SkinSure, a skin disease screening \
tool validated for Fitzpatrick skin types IV-VI. You interpret the structured screening \
results you are given and explain them to a patient in plain language.

Rules:
- You support a clinician; you never diagnose and never prescribe.
- Base everything only on the data provided. Do not invent findings.
- If the models disagree (low agreement) or the explainability data flags an artifact, \
say so in confidence_caveats and lower your confidence_level.
- If you set severity to "urgent", you MUST set immediate_referral to true.
- When unsure between two severities, choose the more cautious one.

Respond with a single JSON object and nothing else - no markdown, no code fences, no \
preamble. Use exactly these keys:
{
  "severity": "routine" | "monitor" | "urgent",
  "summary": string,
  "recommended_action": string,
  "confidence_level": "high" | "moderate" | "low",
  "confidence_caveats": [string],
  "recapture_needed": boolean,
  "immediate_referral": boolean
}"""


def build_context(
    predictions: Predictions,
    patient: PatientMetadata,
    xai_summary: XAISummary,
) -> str:
    """Format the inputs as text for the LLM. Pure formatting - no interpretation."""
    location = patient.patient_location or "not provided"
    return (
        "PATIENT\n"
        f"- Age: {patient.patient_age}\n"
        f"- Fitzpatrick type: {patient.fitzpatrick_type.value}\n"
        f"- Sex: {patient.patient_sex.value}\n"
        f"- Location: {location}\n"
        f"- Season: {patient.current_season.value if patient.current_season else 'not provided'}\n"
        f"- Body area: {patient.body_area.value}\n"
        f"- Symptom duration: {patient.symptom_duration_days} days\n"
        f"- Medical history: {', '.join(patient.medical_history) or 'none reported'}\n"
        f"- Spreading rapidly (patient-reported): {'yes' if patient.is_spreading_rapidly else 'no'}\n"
        "\n"
        "MODEL PREDICTION\n"
        f"- Primary: {predictions.primary_label.value} ({predictions.primary_confidence:.1%})\n"
        f"- Secondary: {predictions.secondary_label.value} ({predictions.secondary_confidence:.1%})\n"
        f"- Model agreement: {predictions.agreement_score:.2f} ({predictions.confidence_tier()})\n"
        "\n"
        "EXPLAINABILITY (Grad-CAM)\n"
        f"- Focus: {xai_summary.focus_description.value}\n"
        f"- Boundary alignment: {xai_summary.boundary_alignment.value}\n"
        f"- Artifact flagged: {'yes' if xai_summary.artifact_flag else 'no'}\n"
    )


def _parse(raw: str) -> ClinicalSummary:
    try:
        return ClinicalSummary(**json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        # pydantic.ValidationError is a ValueError, so schema failures land here too
        raise ValueError(f"Could not parse summarizer response: {exc}\nRaw response:\n{raw}") from exc


def _post_process(
    summary: ClinicalSummary,
    predictions: Predictions,
    xai_summary: XAISummary,
) -> ClinicalSummary:
    """Deterministic rules applied after the LLM - Python, not the LLM, has the last word."""
    if xai_summary.artifact_flag:
        summary.confidence_level = _TIER_DOWN[summary.confidence_level]
        summary.recapture_needed = True

    if predictions.confidence_tier() == "low" and not any(
        "model disagreement" in c.lower() for c in summary.confidence_caveats
    ):
        summary.confidence_caveats.append(_DISAGREEMENT_CAVEAT)

    return summary


async def summarize(
    predictions: Predictions,
    patient: PatientMetadata,
    xai_summary: XAISummary,
    client: anthropic.AsyncAnthropic | None = None,
) -> ClinicalSummary:
    """Call Claude with the formatted context and return a post-processed ClinicalSummary.

    `client` is injectable so tests can pass a fake instead of hitting the API.
    """
    cfg = get_settings().agents.summarizer
    client = client or anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = await client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_context(predictions, patient, xai_summary)}],
    )

    raw = "".join(block.text for block in response.content if block.type == "text").strip()
    summary = _parse(raw)
    return _post_process(summary, predictions, xai_summary)
