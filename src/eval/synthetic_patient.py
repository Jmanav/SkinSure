"""Deterministic, seeded generator of valid PatientMetadata for replay evals.

Every generated patient passes the Fitzpatrick IV–VI gate and all schema validators.
Names are deliberately distinctive (ZZTEST_ prefix) so leakage greps are trivial.
The generator is pure — no I/O, no randomness beyond the seed.
"""
from __future__ import annotations

import random
from typing import Iterator

from api.schemas.patient import BodyArea, FitzpatrickType, PatientMetadata, Season, Sex

_FITZPATRICK_POOL = [FitzpatrickType.IV, FitzpatrickType.V, FitzpatrickType.VI]
_SEX_POOL = [Sex.MALE, Sex.FEMALE, Sex.OTHER]
_BODY_AREAS = list(BodyArea)
_SEASONS = list(Season)

_LOCATIONS = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Kerala",
    "West Bengal", "Rajasthan", "Gujarat", "Uttar Pradesh", None,
]

_HISTORY_POOL = [
    [],
    ["diabetes"],
    ["previous melanoma"],
    ["hypertension", "diabetes"],
    ["eczema history"],
    ["psoriasis"],
    ["immunosuppression"],
    ["family history of skin cancer"],
    ["none reported"],
]

_AGES = [2, 8, 17, 25, 35, 45, 55, 65, 75, 85, 100]
_DURATIONS = [1, 3, 7, 14, 30, 90, 180, 365, 730, 1825, 3650]


def generate_patients(
    n: int,
    *,
    seed: int = 42,
    spreading_rate: float = 0.15,
) -> list[PatientMetadata]:
    """Generate n valid PatientMetadata instances, deterministically."""
    rng = random.Random(seed)
    patients: list[PatientMetadata] = []

    for i in range(n):
        patients.append(PatientMetadata(
            patient_id=i,
            patient_name=f"ZZTEST_Patient_{i:04d}",
            patient_age=rng.choice(_AGES),
            patient_sex=rng.choice(_SEX_POOL),
            fitzpatrick_type=rng.choice(_FITZPATRICK_POOL),
            patient_location=rng.choice(_LOCATIONS),
            current_season=rng.choice(_SEASONS + [None]),
            body_area=rng.choice(_BODY_AREAS),
            symptom_duration_days=rng.choice(_DURATIONS),
            medical_history=list(rng.choice(_HISTORY_POOL)),
            is_spreading_rapidly=rng.random() < spreading_rate,
        ))

    return patients


def generate_patients_iter(
    *,
    seed: int = 42,
    spreading_rate: float = 0.15,
) -> Iterator[PatientMetadata]:
    """Infinite iterator of valid patients, for pairing with an image list of any length."""
    rng = random.Random(seed)
    i = 0
    while True:
        yield PatientMetadata(
            patient_id=i,
            patient_name=f"ZZTEST_Patient_{i:04d}",
            patient_age=rng.choice(_AGES),
            patient_sex=rng.choice(_SEX_POOL),
            fitzpatrick_type=rng.choice(_FITZPATRICK_POOL),
            patient_location=rng.choice(_LOCATIONS),
            current_season=rng.choice(_SEASONS + [None]),
            body_area=rng.choice(_BODY_AREAS),
            symptom_duration_days=rng.choice(_DURATIONS),
            medical_history=list(rng.choice(_HISTORY_POOL)),
            is_spreading_rapidly=rng.random() < spreading_rate,
        )
        i += 1
