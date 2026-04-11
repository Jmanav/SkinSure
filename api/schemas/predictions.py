from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Predictions(BaseModel):
    """Raw output from the ensemble model."""

    primary_label: str = Field(..., description="Top predicted disease class label")
    primary_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the top prediction")
    secondary_label: str | None = Field(default=None, description="Second-ranked disease class label, if applicable")
    secondary_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score for the second prediction"
    )
    agreement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of ensemble models that agree on the top prediction (0.0–1.0)",
    )

    def confidence_tier(self) -> Literal["high", "moderate", "low"]:
        """Map primary_confidence to a human-readable tier.

        Thresholds:
          high     ≥ 0.75
          moderate ≥ 0.50
          low      < 0.50
        """
        if self.primary_confidence >= 0.75:
            return "high"
        if self.primary_confidence >= 0.50:
            return "moderate"
        return "low"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "primary_label": "Tinea versicolor",
                    "primary_confidence": 0.87,
                    "secondary_label": "Vitiligo",
                    "secondary_confidence": 0.09,
                    "agreement_score": 1.0,
                }
            ]
        }
    }


class XAISummary(BaseModel):
    """Structured summary of xAI heatmap output passed to the clinical summarizer."""

    focus_description: str = Field(
        ...,
        description=(
            "Plain-language description of the image region the ensemble attended to most "
            "(e.g. 'central lesion body with high activation on hypopigmented patches')"
        ),
    )
    boundary_alignment: str = Field(
        ...,
        description=(
            "Assessment of whether heatmap activation aligns with the visible lesion boundary "
            "(e.g. 'well-aligned', 'partial — activation extends into healthy skin', 'poor')"
        ),
    )
    artifact_flag: bool = Field(
        ...,
        description="True when the heatmap or input image contains artifacts that may have influenced the prediction",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "focus_description": "High activation concentrated on hypopigmented patches across the upper back; lesion centre dominates.",
                    "boundary_alignment": "well-aligned",
                    "artifact_flag": False,
                }
            ]
        }
    }


class ClinicalSummary(BaseModel):
    primary_condition: str = Field(
        ..., description="Most likely condition identified by the ensemble (e.g. 'Tinea versicolor')"
    )
    confidence_level: Literal["high", "moderate", "low"]
    model_agreement: bool = Field(
        ..., description="True when all three ensemble models agree on the top prediction"
    )
    patient_summary: str = Field(
        ..., description="2–3 sentence plain-language explanation written for the patient"
    )
    severity: Literal["mild", "moderate", "urgent"]
    urgency_reasoning: str = Field(
        ..., description="Explanation of why this severity level was assigned"
    )
    contributing_factors: list[str] = Field(
        ..., description="Contextual factors that influenced the assessment (e.g. 'monsoon season', 'history of diabetes')"
    )
    differential_notes: str = Field(
        ..., description="Brief notes on alternative conditions considered and why they were ranked lower"
    )
    recommended_action: Literal["self_care", "schedule_dermatologist", "immediate_referral"]
    action_details: str = Field(
        ..., description="Specific next steps the patient should take based on the recommended action"
    )
    confidence_caveats: list[str] = Field(
        ..., description="Explicit caveats about uncertainty (e.g. 'image quality limited heatmap reliability')"
    )
    recapture_needed: bool = Field(
        ..., description="True when image quality or artifact flags indicate the photo should be retaken"
    )

    @model_validator(mode="after")
    def urgent_requires_immediate_referral(self) -> ClinicalSummary:
        if self.severity == "urgent" and self.recommended_action != "immediate_referral":
            raise ValueError(
                f"Safety constraint violated: severity is 'urgent' but recommended_action is "
                f"'{self.recommended_action}'. An urgent case must always map to "
                f"'immediate_referral' — the LLM or triage layer must not downgrade this."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "primary_condition": "Tinea versicolor",
                    "confidence_level": "high",
                    "model_agreement": True,
                    "patient_summary": (
                        "The screening is consistent with tinea versicolor, a common fungal skin condition "
                        "that causes patchy discolouration. This is frequently seen during monsoon season in "
                        "India due to increased humidity. It is generally not harmful, but a pharmacist or "
                        "doctor can confirm and recommend appropriate antifungal treatment."
                    ),
                    "severity": "mild",
                    "urgency_reasoning": "Condition is non-spreading, non-inflammatory, and has no features associated with systemic illness.",
                    "contributing_factors": ["monsoon season", "high ambient humidity", "upper back location"],
                    "differential_notes": (
                        "Vitiligo was considered but depigmentation pattern and border characteristics are "
                        "more consistent with tinea versicolor. Pityriasis rosea ranked third due to absence of herald patch."
                    ),
                    "recommended_action": "self_care",
                    "action_details": (
                        "Visit a nearby pharmacy for an OTC antifungal cream (e.g. clotrimazole or ketoconazole). "
                        "Keep the area dry and clean. If the rash spreads or does not improve within 4 weeks, "
                        "consult a dermatologist."
                    ),
                    "confidence_caveats": [],
                    "recapture_needed": False,
                }
            ]
        }
    }
