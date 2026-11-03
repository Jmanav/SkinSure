"""Tests for the synthetic patient generator (src/eval/synthetic_patient.py)."""
from __future__ import annotations

import pytest

from api.schemas.patient import FitzpatrickType, PatientMetadata
from src.eval.synthetic_patient import generate_patients, generate_patients_iter


class TestGeneratePatients:
    def test_returns_requested_count(self):
        assert len(generate_patients(10)) == 10
        assert len(generate_patients(0)) == 0

    def test_all_pass_schema_validation(self):
        for p in generate_patients(50, seed=99):
            assert isinstance(p, PatientMetadata)

    def test_only_fitzpatrick_iv_vi(self):
        types = {p.fitzpatrick_type for p in generate_patients(200)}
        assert types <= {FitzpatrickType.IV, FitzpatrickType.V, FitzpatrickType.VI}
        assert len(types) == 3

    def test_deterministic_with_same_seed(self):
        a = generate_patients(20, seed=7)
        b = generate_patients(20, seed=7)
        for pa, pb in zip(a, b):
            assert pa.model_dump() == pb.model_dump()

    def test_different_seeds_differ(self):
        a = generate_patients(20, seed=1)
        b = generate_patients(20, seed=2)
        assert any(
            pa.model_dump() != pb.model_dump() for pa, pb in zip(a, b)
        )

    def test_names_are_greppable(self):
        for p in generate_patients(10):
            assert p.patient_name.startswith("ZZTEST_")

    def test_spreading_rate_zero_means_none_spread(self):
        patients = generate_patients(100, spreading_rate=0.0)
        assert not any(p.is_spreading_rapidly for p in patients)

    def test_spreading_rate_one_means_all_spread(self):
        patients = generate_patients(100, spreading_rate=1.0)
        assert all(p.is_spreading_rapidly for p in patients)

    def test_age_within_bounds(self):
        for p in generate_patients(200):
            assert 0 <= p.patient_age <= 120

    def test_duration_non_negative(self):
        for p in generate_patients(200):
            assert p.symptom_duration_days >= 0


class TestGeneratePatientsIter:
    def test_yields_valid_patients(self):
        it = generate_patients_iter(seed=42)
        for _, p in zip(range(30), it):
            assert isinstance(p, PatientMetadata)

    def test_deterministic(self):
        a = [p for _, p in zip(range(20), generate_patients_iter(seed=5))]
        b = [p for _, p in zip(range(20), generate_patients_iter(seed=5))]
        for pa, pb in zip(a, b):
            assert pa.model_dump() == pb.model_dump()

    def test_ids_are_sequential(self):
        patients = [p for _, p in zip(range(10), generate_patients_iter())]
        assert [p.patient_id for p in patients] == list(range(10))
