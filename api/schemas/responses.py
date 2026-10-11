from pydantic import BaseModel, Field

from api.schemas.predictions import ClinicalSummary, Predictions, ScreeningStatus


class PredictResponse(BaseModel):
    ''' What POST /predict returns. Deliberately omits patient_name, the raw XAI data and the
    internal triage rule log; predictions are only included when a result is actually shown.'''
    status: ScreeningStatus
    summary: ClinicalSummary | None = None          # only when status == completed
    predictions: Predictions | None = None          # only when status == completed
    messages: list[str] = Field(default_factory=list)  # why a photo was rejected / needs a retake
    warnings: list[str] = Field(default_factory=list)
    escalated: bool = False                         # a safety rule raised the LLM's severity
    treatment_advice_allowed: bool = False
