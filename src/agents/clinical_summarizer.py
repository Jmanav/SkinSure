from __future__ import annotations

import json

import anthropic

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import ClinicalSummary, Predictions, XAISummary

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1000

_TIER_DOWN: dict[str, str] = {"high": "moderate", "moderate": "low", "low": "low"}

_SYSTEM_PROMPT = """\
You are the clinical summarizer for SkinSure, an AI skin-disease screening tool \
for patients in India with Fitzpatrick IV–VI skin tones.

Your task is to produce a structured clinical summary from model predictions, \
patient metadata, and explainability analysis. You will receive a context block \
and must return a single valid JSON object that conforms exactly to the \
ClinicalSummary schema below — no markdown, no preamble, no trailing text.

=== ClinicalSummary JSON schema ===
{
  "primary_condition": string,
  "confidence_level": "high" | "moderate" | "low",
  "model_agreement": boolean,
  "patient_summary": string (2–3 sentences, plain language for the patient),
  "severity": "mild" | "moderate" | "urgent",
  "urgency_reasoning": string,
  "contributing_factors": [string, ...],
  "differential_notes": string,
  "recommended_action": "self_care" | "schedule_dermatologist" | "immediate_referral",
  "action_details": string,
  "confidence_caveats": [string, ...],
  "recapture_needed": boolean
}

=== Rules you must follow ===
1. Never state a diagnosis as certain. Use "the screening suggests" or \
"consistent with" — never "you have" or "this is".
2. If model agreement is below 70%, add an explicit uncertainty string to \
confidence_caveats.
3. If xAI artifacts are flagged, set recapture_needed to true and note it in \
confidence_caveats.
4. Never set recommended_action to "self_care" when severity is "urgent". \
An urgent case must always map to "immediate_referral".
5. Always factor in the patient's region and current season when assessing \
contributing_factors. India-specific epidemiology (monsoon fungal outbreaks, \
regional prevalence of tinea versicolor, vitiligo, etc.) must inform the \
contributing_factors and differential_notes fields.
6. Set model_agreement to true only when agreement_score >= 0.70.
7. Output valid JSON only — no other text.\
"""


def build_context(
    predictions: Predictions,
    metadata: PatientMetadata,
    xai_summary: XAISummary,
) -> str:
    """
    Assemble a structured prompt context string from three input streams.
    Pure Python — no LLM calls, no interpretation.

    Returns a formatted string with three labelled sections:
      1. Model Predictions
      2. Patient Context
      3. xAI Analysis
    """
    tier = predictions.confidence_tier()
    agreement_pct = round(predictions.agreement_score * 100, 1)

    secondary_label = predictions.secondary_label or "N/A"
    secondary_conf = (
        f"{predictions.secondary_confidence:.1%}" if predictions.secondary_confidence is not None else "N/A"
    )

    history = metadata.medical_history
    history_str = ", ".join(history) if history else "None reported"

    artifact_note = (
        "Yes — recommend recapture" if xai_summary.artifact_flag else "None detected"
    )

    return (
        "=== Model Predictions ===\n"
        f"Primary condition   : {predictions.primary_label}\n"
        f"Primary confidence  : {predictions.primary_confidence:.1%}  [{tier}]\n"
        f"Secondary condition : {secondary_label}\n"
        f"Secondary confidence: {secondary_conf}\n"
        f"Model agreement     : {agreement_pct}%\n"
        "\n"
        "=== Patient Context ===\n"
        f"Age                 : {metadata.age}\n"
        f"Sex                 : {metadata.sex}\n"
        f"Fitzpatrick type    : {metadata.fitzpatrick}\n"
        f"Region              : {metadata.region}\n"
        f"Season              : {metadata.current_season}\n"
        f"Body area           : {metadata.body_area}\n"
        f"Symptom duration    : {metadata.symptom_duration}\n"
        f"Medical history     : {history_str}\n"
        "\n"
        "=== xAI Analysis ===\n"
        f"Focus description   : {xai_summary.focus_description}\n"
        f"Boundary alignment  : {xai_summary.boundary_alignment}\n"
        f"Artifact detected   : {artifact_note}\n"
    )


async def summarize(
    predictions: Predictions,
    metadata: PatientMetadata,
    xai_summary: XAISummary,
) -> ClinicalSummary:
    """
    Call the Anthropic API with the assembled context and return a validated
    ClinicalSummary.

    Post-processing (applied in Python after the LLM response):
    - If xai_summary.artifact_flag: downgrade confidence_level one tier and
      ensure recapture_needed is True.
    - If predictions.agreement_score < 0.70 and no caveat already mentions
      "model disagreement": append the standard disagreement caveat.
    """
    context = build_context(predictions, metadata, xai_summary)

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = response.content[0].text.strip()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Clinical summarizer received non-JSON response from LLM.\n"
            f"JSONDecodeError: {exc}\n"
            f"Raw response:\n{raw_text}"
        ) from exc

    summary = ClinicalSummary.model_validate(payload)

    # ── Post-processing ────────────────────────────────────────────────────
    if xai_summary.artifact_flag:
        summary.confidence_level = _TIER_DOWN[summary.confidence_level]
        summary.recapture_needed = True

    if predictions.agreement_score < 0.70:
        _disagreement_caveat = (
            "Model agreement below 70% — treat confidence tier as indicative only"
        )
        already_noted = any(
            "model disagreement" in c.lower() for c in summary.confidence_caveats
        )
        if not already_noted:
            summary.confidence_caveats.append(_disagreement_caveat)

    return summary
