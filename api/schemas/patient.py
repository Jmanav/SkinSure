from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Optional

#the value is both str and enum
# someone can pass "VII" or "four" and your code breaks silently.
class FitzpatrickType(str,Enum):
    I = 'I'
    II = 'II'
    III = 'III'
    IV =  'IV'
    V = 'V'
    VI = 'VI'

class PatientMetadata(BaseModel):
    ''' The schema for the patient metadata that is required for the model to make predictions.'''
    patient_id : int
    patient_name : str
    patient_age : int
    fitzpatrick_type : FitzpatrickType
    patient_location : str | None = None #optional

    @field_validator("fitzpatrick_type")
    @classmethod #why do we need a class method here? because it needs to run before the actual instance of PatientMetadata is created, so it can't be an instance method. It needs to be a class method that can be called without an instance of the class.
    def validate_fitzpatrick(cls,v):
        if v in (FitzpatrickType.I, FitzpatrickType.II, FitzpatrickType.III):
            raise ValueError(
                f"SkinSure is validated only for Fitzpatrick types IV - VI"
                f"received:{v}"
            )
        return v
