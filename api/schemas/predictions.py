#why do we inherit from BaseModel and not just simply use a class with attributes?
from pydantic import BaseModel, model_validator
 
class Predictions(BaseModel):
    ''' The schema must match what the ensemble actually produces, not what I assumed.'''
    primary_label : str
    primary_confidence : float
    secondary_label : str
    secondary_confidence: float 
    agreement_score : float

    def confidence_tier(self) -> str:
        if self.agreement_score >= 85:
            return "high"
        elif self.agreement_score >= 75:
            return "moderate"
        else: return "low"

class XAISummary(BaseModel):
    ''' The schema for the XAI summary that is produced by the model.'''
    summary : str
    flag_artifacts : bool
    flag_blur : bool    

class ClinicalSummary(BaseModel):
    ''' The schema for the clinical summary that is produced by the model.'''
    severity : str
    confidence_level : float
    summary : str

    @model_validator(mode="after")
    def validate_urgency(self):
        if self.severity == "urgent" and self.confidence_level < 0.85:
            raise ValueError(
                f"Severity is 'urgent' but confidence level is too low, received {self.severity}"
            )
        return self
    