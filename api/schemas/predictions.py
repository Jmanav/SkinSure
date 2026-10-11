#why do we inherit from BaseModel and not just simply use a class with attributes?
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class DiseaseClass(str, Enum):
    ''' The 5 classes the ensemble was trained on. Values are the exact strings stored in each
    checkpoint's "classes" (alphabetical = model index order).'''
    BASAL_CELL_CARCINOMA = "Basal_Cell_Carcinoma"
    ECZEMA = "Eczema"
    KERATOSIS = "Keratosis"
    MELANOMA = "Melanoma"
    TINEA = "Tinea"


class Predictions(BaseModel):
    ''' The schema must match what the ensemble actually produces, not what I assumed.'''
    # Enum-typed labels: a typo or an unknown class fails here, instead of silently
    # slipping past the label comparisons in severity_triage.
    primary_label : DiseaseClass
    primary_confidence : float = Field(ge=0.0, le=1.0)
    secondary_label : DiseaseClass
    secondary_confidence: float = Field(ge=0.0, le=1.0)
    agreement_score : float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_top2(self):
        if self.secondary_label == self.primary_label:
            raise ValueError("secondary_label must differ from primary_label")
        if self.secondary_confidence > self.primary_confidence:
            raise ValueError("secondary_confidence cannot exceed primary_confidence")
        return self

    def confidence_tier(self) -> str:
        # agreement_score is a fraction of 3 models: 0.0 / 0.33 / 0.67 / 1.0.
        # So: all three agree -> high, two agree -> moderate, otherwise low.
        if self.agreement_score >= 0.99:
            return "high"
        elif self.agreement_score >= 0.66:
            return "moderate"
        else: return "low"


class FocusRegion(str, Enum):
    ''' Where the Grad-CAM heat is concentrated.'''
    LESION_CENTRE = "lesion_centre"
    LESION_BORDER = "lesion_border"
    DIFFUSE = "diffuse"                  # spread over the image, no clear focus
    BACKGROUND_SKIN = "background_skin"  # healthy skin away from the lesion
    ARTIFACT = "artifact"                # ruler / ink / bubble / hair


class BoundaryAlignment(str, Enum):
    ''' How well the attended region lines up with the lesion border.'''
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"


class XAISummary(BaseModel):
    ''' Grad-CAM output reduced to plain facts the summarizer can read (no heatmap arrays).'''
    focus_description : FocusRegion
    boundary_alignment : BoundaryAlignment
    artifact_flag : bool         # True if the model attended to a ruler/ink/bubble etc.

    @model_validator(mode="after")
    def validate_artifact_consistency(self):
        # A focus on an artifact must raise the flag, otherwise the summarizer's
        # confidence downgrade / recapture rule would silently not fire.
        if self.focus_description == FocusRegion.ARTIFACT and not self.artifact_flag:
            raise ValueError("focus_description 'artifact' requires artifact_flag=True")
        return self


# str + Enum: members compare equal to their string value and serialise as plain JSON strings,
# and pydantic rejects anything outside the set (so the LLM can't invent a new severity).
class Severity(str, Enum):
    ROUTINE = "routine"
    MONITOR = "monitor"
    URGENT = "urgent"


class ScreeningStatus(str, Enum):
    COMPLETED = "completed"  # a summary is available
    REJECTED = "rejected"    # image failed the quality gate; the model never ran
    BLOCKED = "blocked"      # model ran but confidence is too low; result withheld, re-capture


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
