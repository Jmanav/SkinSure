from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from api.schemas.patient import BodyArea, FitzpatrickType, PatientMetadata, Season, Sex
from api.schemas.predictions import Predictions, XAISummary
from src.agents.clinical_summarizer import summarize


async def main() -> None:
    load_dotenv()  # puts ANTHROPIC_API_KEY from .env into the environment

    predictions = Predictions(
        primary_label="melanoma",
        primary_confidence=0.62,
        secondary_label="nevus",
        secondary_confidence=0.25,
        agreement_score=0.33,  # only 1 of 3 agrees -> "low" tier -> disagreement caveat
    )
    patient = PatientMetadata(
        patient_id=1,
        patient_name="Test Patient",
        patient_age=45,
        fitzpatrick_type=FitzpatrickType.V,
        patient_sex=Sex.FEMALE,
        patient_location="Pune",
        current_season=Season.MONSOON,
        body_area=BodyArea.ARM,
        symptom_duration_days=60,
        medical_history=["diabetes"],
        is_spreading_rapidly=False,
    )
    xai = XAISummary(
        focus_description="centre of the lesion",
        boundary_alignment="partial",
        artifact_flag=False,
    )

    summary = await summarize(predictions, patient, xai)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
