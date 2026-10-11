from enum import Enum

from pydantic import BaseModel, field_validator


# str + Enum: values are plain strings in JSON, but anything outside the set is rejected
# (someone passing "VII" or "four" fails loudly instead of breaking silently).
class FitzpatrickType(str, Enum):
    I = 'I'
    II = 'II'
    III = 'III'
    IV = 'IV'
    V = 'V'
    VI = 'VI'


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Season(str, Enum):
    SUMMER = "summer"
    MONSOON = "monsoon"
    WINTER = "winter"
    SPRING = "spring"
    AUTUMN = "autumn"


class BodyArea(str, Enum):
    HEAD_NECK = "head_neck"
    TRUNK = "trunk"
    ARM = "arm"
    HAND = "hand"
    LEG = "leg"
    FOOT = "foot"
    GENITAL = "genital"
    OTHER = "other"


class PatientMetadata(BaseModel):
    ''' The schema for the patient metadata that is required for the model to make predictions.'''
    patient_id : int
    patient_name : str
    patient_age : int
    patient_sex : Sex
    fitzpatrick_type : FitzpatrickType
    patient_location : str | None = None  # region, optional
    current_season : Season | None = None
    body_area : BodyArea
    symptom_duration_days : int   # how long the lesion/symptoms have been present
    medical_history : list[str] = []  # e.g. ["diabetes", "previous melanoma"]
    is_spreading_rapidly : bool = False  # patient-reported; feeds the triage override for infections

    @field_validator("fitzpatrick_type")
    @classmethod  # runs on the class during construction, before any instance exists
    def validate_fitzpatrick(cls, v):
        if v in (FitzpatrickType.I, FitzpatrickType.II, FitzpatrickType.III):
            raise ValueError(
                f"SkinSure is validated only for Fitzpatrick types IV - VI, received: {v.value}"
            )
        return v

    @field_validator("patient_age")
    @classmethod
    def validate_age(cls, v):
        if not 0 <= v <= 120:
            raise ValueError("patient_age must be between 0 and 120")
        return v

    @field_validator("symptom_duration_days")
    @classmethod
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError("symptom_duration_days cannot be negative")
        return v
