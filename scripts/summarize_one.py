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


CASES = ("disagreement", "artifact", "urgent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise one hard-coded case through the real Anthropic API."
    )
    parser.add_argument(
        "--case",
        choices=CASES,
        default="disagreement",
        help=(
            "Which scenario to send. 'disagreement': low agreement -> the standard caveat is "
            "appended. 'artifact': artifact_flag=True -> confidence downgraded one tier and "
            "recapture forced. 'urgent': melanoma at 0.91 -> the LLM should return urgent, and "
            "the schema validator requires immediate_referral with it."
        ),
    )
    add_spend_args(parser)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    guard = SpendGuard.from_args(args)  # refuses unless authorised and capped

    load_dotenv()  # puts ANTHROPIC_API_KEY from .env into the environment

    # Each case isolates one behaviour that the fake-client tests pin but cannot prove
    # against a real model: the caveat append, the artifact downgrade, and whether the
    # LLM itself honours the urgent -> immediate_referral rule the validator also enforces.
    if args.case == "disagreement":
        predictions = Predictions(
            primary_label=DiseaseClass.MELANOMA,
            primary_confidence=0.62,
            secondary_label=DiseaseClass.KERATOSIS,
            secondary_confidence=0.25,
            agreement_score=0.33,  # only 1 of 3 agrees -> "low" tier -> disagreement caveat
        )
    elif args.case == "artifact":
        # All three models agree, so the only thing that may lower confidence is the
        # artifact flag -- which makes the downgrade visible rather than confounded.
        predictions = Predictions(
            primary_label=DiseaseClass.ECZEMA,
            primary_confidence=0.88,
            secondary_label=DiseaseClass.TINEA,
            secondary_confidence=0.07,
            agreement_score=1.0,
        )
    else:  # urgent
        predictions = Predictions(
            primary_label=DiseaseClass.MELANOMA,
            primary_confidence=0.91,
            secondary_label=DiseaseClass.KERATOSIS,
            secondary_confidence=0.05,
            agreement_score=1.0,
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
    if args.case == "artifact":
        # The schema validator requires artifact_flag=True when focus is `artifact`.
        xai = XAISummary(
            focus_description=FocusRegion.ARTIFACT,
            boundary_alignment=BoundaryAlignment.POOR,
            artifact_flag=True,
        )
    else:
        xai = XAISummary(
            focus_description=FocusRegion.LESION_CENTRE,
            boundary_alignment=BoundaryAlignment.PARTIAL,
            artifact_flag=False,
        )

    client = _MeteredClient(anthropic.AsyncAnthropic())
    summary = await summarize(predictions, patient, xai, client=client)
    print(f"case: {args.case}")
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
