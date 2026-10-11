#why do we inherit from BaseModel and not just simply use a class with attributes?
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Predictions(BaseModel):
    ''' The schema must match what the ensemble actually produces, not what I assumed.'''
    primary_label : str
    primary_confidence : float
    secondary_label : str
    secondary_confidence: float
    agreement_score : float

    def confidence_tier(self) -> str:
        # agreement_score is a fraction of 3 models: 0.0 / 0.33 / 0.67 / 1.0.
        # So: all three agree -> high, two agree -> moderate, otherwise low.
        if self.agreement_score >= 0.99:
            return "high"
        elif self.agreement_score >= 0.66:
            return "moderate"
        else: return "low"


class XAISummary(BaseModel):
    ''' Grad-CAM output reduced to plain facts the summarizer can read (no heatmap arrays).'''
    focus_description : str      # where the model looked, e.g. "centre of the lesion"
    boundary_alignment : str     # how well the focus lines up with the lesion border
    artifact_flag : bool         # True if the model attended to a ruler/ink/bubble etc.


# str + Enum: members compare equal to their string value and serialise as plain JSON strings,
# and pydantic rejects anything outside the set (so the LLM can't invent a new severity).
class Severity(str, Enum):
    ROUTINE = "routine"
    MONITOR = "monitor"
    URGENT = "urgent"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class ClinicalSummary(BaseModel):
    ''' The structured output of the clinical summarizer agent.'''
    severity : Severity
    summary : str                                   # patient-friendly explanation
    recommended_action : str
    confidence_level : ConfidenceLevel
    confidence_caveats : list[str] = Field(default_factory=list)
    recapture_needed : bool = False
    immediate_referral : bool = False

    @model_validator(mode="after")
    def validate_urgency(self):
        # Safety constraint, second line of defence (the first is the LLM system prompt).
        if self.severity == Severity.URGENT and not self.immediate_referral:
            raise ValueError("severity 'urgent' requires immediate_referral=True")
        return self
