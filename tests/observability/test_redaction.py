"""The PII boundary, tested before any exporter exists.

A span that has been exported cannot be un-exported, so these tests run against the
pure functions rather than waiting for a tracing backend. The important ones are:

  * `test_no_patient_name_anywhere_in_attributes` and its str/repr variants -- the name
    must not appear in a key, a value, or the repr of the whole attribute dict.
  * `test_every_patient_metadata_field_is_decided` -- the regression guard that matters.
    It fails when someone adds a field to `PatientMetadata` without deciding whether it
    is safe to trace. Without it, a new field is silently absent today and silently
    *present* the moment somebody "helpfully" switches the allowlist to a denylist.
"""
from __future__ import annotations

import hashlib

import pytest

from api.schemas.patient import PatientMetadata
from src.observability.redaction import (
    MAX_LOCATION,
    MAX_TEXT,
    NEVER_EMIT,
    coarsen_location,
    image_fingerprint,
    pseudonymise,
    redact,
    salt_source,
    span_attributes,
    truncate,
)

SECRET_NAME = "Zzyzx Qwertyuiop"          # distinctive, so a leak is unmistakable


# --- the name must never appear, in any form -----------------------------------------

def test_no_patient_name_anywhere_in_attributes(make_patient):
    attrs = span_attributes(make_patient(patient_name=SECRET_NAME))

    assert "patient_name" not in attrs
    assert not any("name" in k for k in attrs)
    assert SECRET_NAME not in repr(attrs)
    for part in SECRET_NAME.split():
        assert part not in repr(attrs)


def test_name_absent_even_when_it_looks_like_other_data(make_patient):
    """A name that collides with a traced value must still not appear as a name field."""
    attrs = span_attributes(make_patient(patient_name="45", patient_age=45))
    assert "patient_name" not in attrs
    assert attrs["patient.age"] == 45          # the age is still traced, as itself


def test_redact_blanks_never_emit_keys():
    payload = {k: "sensitive" for k in NEVER_EMIT}
    out = redact(payload)
    for key in NEVER_EMIT:
        assert out[key] == "[redacted]"
        assert "sensitive" not in repr(out[key])


# --- patient_id is pseudonymous, not identifying --------------------------------------

def test_patient_id_is_not_emitted_raw(make_patient):
    attrs = span_attributes(make_patient(patient_id=424242))
    assert "patient_id" not in attrs
    assert "424242" not in repr(attrs)
    assert attrs["patient.pid"].startswith("pid_")


def test_pseudonymise_is_stable_and_distinct():
    assert pseudonymise(7) == pseudonymise(7)          # groupable
    assert pseudonymise(7) != pseudonymise(8)          # distinguishable


def test_pseudonymise_is_salted(monkeypatch):
    """Different deployments must not produce joinable tokens."""
    monkeypatch.setenv("SKINSURE_TRACE_SALT", "salt-a")
    a = pseudonymise(7)
    monkeypatch.setenv("SKINSURE_TRACE_SALT", "salt-b")
    b = pseudonymise(7)
    assert a != b


def test_pseudonymise_is_not_a_bare_hash_of_the_id(monkeypatch):
    """A bare sha256 of a small int is reversible by brute force; this must not be one."""
    monkeypatch.setenv("SKINSURE_TRACE_SALT", "some-real-salt")
    token = pseudonymise(7)
    bare = hashlib.sha256(b"7").hexdigest()
    assert bare[:16] not in token


def test_salt_source_flags_the_dev_fallback(monkeypatch):
    monkeypatch.delenv("SKINSURE_TRACE_SALT", raising=False)
    assert "NOT secure" in salt_source()
    monkeypatch.setenv("SKINSURE_TRACE_SALT", "x")
    assert salt_source() == "env"


# --- free text and location -----------------------------------------------------------

def test_medical_history_content_never_traced(make_patient):
    attrs = span_attributes(make_patient(
        medical_history=["my brother Dr Rao at Apollo treated this", "diabetes"]))
    assert attrs["patient.medical_history_count"] == 2
    assert "Rao" not in repr(attrs)
    assert "Apollo" not in repr(attrs)


def test_location_is_coarsened(make_patient):
    attrs = span_attributes(make_patient(patient_location="Bandra West, Mumbai, Maharashtra"))
    assert attrs["patient.location_coarse"] == "Bandra West"
    assert "Maharashtra" not in repr(attrs)


def test_location_absent_when_not_given(make_patient):
    assert "patient.location_coarse" not in span_attributes(make_patient())


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("Mumbai", "Mumbai"),
    ("Mumbai, Maharashtra", "Mumbai"),
    ("  Pune  , MH", "Pune"),
])
def test_coarsen_location_cases(value, expected):
    assert coarsen_location(value) == expected


def test_coarsen_location_truncates():
    assert len(coarsen_location("x" * 500)) <= MAX_LOCATION


def test_truncate_marks_clipping():
    out = truncate("y" * (MAX_TEXT + 50))
    assert len(out) == MAX_TEXT
    assert out.endswith("…")


def test_truncate_leaves_short_text_alone():
    assert truncate("short") == "short"


# --- images: fingerprint, never content -----------------------------------------------

def test_image_fingerprint_carries_no_pixels():
    data = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8
    fp = image_fingerprint(data, width=640, height=480)

    assert fp["image.sha256"] == hashlib.sha256(data).hexdigest()
    assert fp["image.bytes"] == len(data)
    assert fp["image.short_side"] == 480
    # no attribute may carry raw bytes
    for value in fp.values():
        assert not isinstance(value, (bytes, bytearray))


def test_image_fingerprint_is_stable_for_the_same_upload():
    data = b"same-bytes"
    assert image_fingerprint(data)["image.sha256"] == image_fingerprint(data)["image.sha256"]


def test_redact_replaces_raw_bytes_with_a_digest():
    out = redact({"blob": b"secret-image-bytes"})
    assert "secret-image-bytes" not in repr(out)
    assert "bytes, sha256=" in out["blob"]


# --- the regression guard that actually protects us -----------------------------------

# Every field on PatientMetadata must be in exactly one of these sets. Adding a field to
# the schema without classifying it here fails the test below -- which is the point.
TRACED = {
    "patient_id",                 # pseudonymised to patient.pid
    "patient_age",
    "patient_sex",
    "fitzpatrick_type",
    "body_area",
    "symptom_duration_days",
    "is_spreading_rapidly",
    "current_season",
    "patient_location",           # coarsened
    "medical_history",            # reduced to a count
}
WITHHELD = {
    "patient_name",               # dropped entirely; never hashed either
}


def test_every_patient_metadata_field_is_decided():
    """Fails when a new PatientMetadata field has not been classified.

    This is the guard that makes the allowlist trustworthy over time. A new field is
    excluded from traces by default, but 'excluded because nobody looked' is not a
    decision -- so force the decision here.
    """
    fields = set(PatientMetadata.model_fields)
    classified = TRACED | WITHHELD
    unclassified = fields - classified
    assert not unclassified, (
        f"PatientMetadata gained field(s) {sorted(unclassified)} that are not classified "
        "in tests/observability/test_redaction.py. Decide whether each is safe to put on "
        "a span, then add it to TRACED (and to span_attributes()) or to WITHHELD."
    )
    assert not classified - fields, (
        f"these classified fields no longer exist on PatientMetadata: "
        f"{sorted(classified - fields)}"
    )


def test_withheld_fields_are_in_never_emit_or_simply_absent(make_patient):
    attrs = span_attributes(make_patient(patient_name=SECRET_NAME))
    for field in WITHHELD:
        assert field not in attrs


def test_traced_fields_actually_appear(make_patient):
    """The allowlist must not silently stop emitting something we rely on."""
    attrs = span_attributes(make_patient(
        patient_location="Mumbai", current_season=None, medical_history=["x"]))
    for key in ("patient.pid", "patient.age", "patient.sex", "patient.fitzpatrick",
                "patient.body_area", "patient.symptom_duration_days",
                "patient.is_spreading_rapidly", "patient.medical_history_count"):
        assert key in attrs, f"{key} missing from span attributes"


def test_attributes_are_otel_safe_types(make_patient):
    """OTel accepts str/bool/int/float (and homogeneous sequences); anything else is
    dropped or stringified by the SDK. Catch a stray enum or dict here instead."""
    attrs = span_attributes(make_patient(patient_location="Mumbai"))
    for key, value in attrs.items():
        assert isinstance(value, (str, bool, int, float)), f"{key} is {type(value)}"


def test_enums_are_reduced_to_their_values(make_patient):
    attrs = span_attributes(make_patient())
    assert attrs["patient.fitzpatrick"] == "V"
    assert not repr(attrs).count("FitzpatrickType.")


# --- the dict/model equivalence that keeps this module dependency-free ----------------

def test_span_attributes_accepts_a_plain_dict(make_patient):
    patient = make_patient(patient_location="Mumbai")
    from_model = span_attributes(patient)
    from_dict = span_attributes(patient.model_dump())
    assert from_model == from_dict


def test_redact_recurses_into_nested_structures():
    out = redact({"outer": {"patient_id": 9, "patient_name": "X", "ok": "fine"}})
    inner = out["outer"]
    assert inner["patient_pid"].startswith("pid_")
    assert inner["patient_name"] == "[redacted]"
    assert inner["ok"] == "fine"


def test_redact_preserves_list_shape():
    out = redact(["a", {"patient_name": "X"}])
    assert isinstance(out, list)
    assert out[1]["patient_name"] == "[redacted]"
