from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PatientMetadata(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age in years (1–120)")
    sex: Literal["male", "female", "other"]
    fitzpatrick: Literal["IV", "V", "VI"] = Field(
        ..., description="Fitzpatrick skin phototype — SkinSure gates to IV–VI only"
    )
    region: str = Field(
        ..., description="Indian state or district of the patient (e.g. 'Tamil Nadu', 'Patna, Bihar')"
    )
    current_season: Literal["summer", "monsoon", "winter", "spring"]
    body_area: str = Field(..., description="Body area where the lesion is located (e.g. 'left forearm')")
    symptom_duration: str = Field(
        ..., description="How long symptoms have been present (e.g. '3 weeks', '2 months')"
    )
    medical_history: list[str] | None = Field(
        default=None,
        description="Relevant past conditions or medications (e.g. ['Type 2 diabetes', 'topical steroid use'])",
    )

    @field_validator("age", mode="before")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (1 <= v <= 120):
            raise ValueError(f"age must be between 1 and 120, got {v}")
        return v

    @field_validator("fitzpatrick", mode="before")
    @classmethod
    def validate_fitzpatrick(cls, v: str) -> str:
        allowed = {"IV", "V", "VI"}
        if v not in allowed:
            raise ValueError(
                f"SkinSure only supports Fitzpatrick phototypes IV, V, and VI (darker skin tones). "
                f"Received '{v}'. If the patient has a lighter skin tone, they are outside this "
                f"system's validated scope."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 34,
                    "sex": "female",
                    "fitzpatrick": "V",
                    "region": "Coimbatore, Tamil Nadu",
                    "current_season": "monsoon",
                    "body_area": "upper back",
                    "symptom_duration": "3 weeks",
                    "medical_history": ["Type 2 diabetes", "topical corticosteroid use (6 months)"],
                }
            ]
        }
    }
