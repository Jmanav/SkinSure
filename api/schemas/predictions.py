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