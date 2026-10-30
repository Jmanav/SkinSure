from __future__ import annotations

import argparse
import asyncio

import anthropic
from dotenv import load_dotenv

from api.schemas.patient import BodyArea, FitzpatrickType, PatientMetadata, Season, Sex
from api.schemas.predictions import (
    BoundaryAlignment,
    DiseaseClass,
    FocusRegion,
    Predictions,
    XAISummary,
)
from src.agents.clinical_summarizer import summarize

if __package__:  # imported as scripts.summarize_one (tests)
    from ._guard import SpendGuard, SpendLimitReached, SpendRefused, add_spend_args
else:  # run as `python scripts/summarize_one.py`, so scripts/ is on sys.path
    from _guard import SpendGuard, SpendLimitReached, SpendRefused, add_spend_args


class _MeteredClient:
    """Wraps the real client so the guard can price the call.

    summarize() returns a ClinicalSummary and drops response.usage, so the token
    counts are not visible to the caller. Rather than change that return contract,
    this records the usage on the way through.
    """

    def __init__(self, inner: anthropic.AsyncAnthropic):
        self._inner = inner
        self.usage = None
        self.messages = type("_M", (), {"create": self._create})()

    async def _create(self, **kwargs):
        response = await self._inner.messages.create(**kwargs)
        self.usage = response.usage
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise one hard-coded case through the real Anthropic API."
    )
    add_spend_args(parser)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    guard = SpendGuard.from_args(args)  # refuses unless authorised and capped

    load_dotenv()  # puts ANTHROPIC_API_KEY from .env into the environment

    predictions = Predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.62,
        secondary_label=DiseaseClass.KERATOSIS,
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
        focus_description=FocusRegion.LESION_CENTRE,
        boundary_alignment=BoundaryAlignment.PARTIAL,
        artifact_flag=False,
    )

    client = _MeteredClient(anthropic.AsyncAnthropic())
    summary = await summarize(predictions, patient, xai, client=client)
    print(summary.model_dump_json(indent=2))

    # After the request, because the cost is only known from the response. One call,
    # so the cap cannot bind mid-run here; the abort is reported rather than raised.
    try:
        cost = guard.record(client.usage)
        print(f"\ncost: ${cost:.6f}  |  {guard.summary()}")
    except SpendLimitReached as exc:
        print(f"\n{exc}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SpendRefused as exc:
        # An expected outcome, not a crash: print the instructions, exit non-zero.
        raise SystemExit(f"refused: {exc}") from None
