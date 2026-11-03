"""Redaction for anything that leaves the process: spans, logs, error bodies.

**This module is the PII boundary.** Everything here is deliberately pure stdlib and has
no OpenTelemetry dependency, so it is fully testable before any exporter exists and
cannot be bypassed by a tracing-backend change.

Why it is written first, before any span is emitted: a span that has already been
exported cannot be un-exported. Every other part of the tracing work is reversible; this
is not. The rule is **allowlist, never denylist** — `span_attributes()` names the fields
it will emit, so a field added to `PatientMetadata` later is excluded by default rather
than leaking until someone notices.

What the project's invariants require (CLAUDE.md "Design invariants"):
  * "Secrets never touch git or images."
  * Uploaded images are never used for training; they exist only for the patient's own
    history. Nothing here writes image content anywhere.

Concretely:
  * `patient_name` is **dropped entirely**. It is never needed to debug a request, and
    `build_context()` already omits it from the LLM prompt. There is no hashed form of
    it either, because a name is low-entropy and a hash would be re-identifiable by
    dictionary attack.
  * `patient_id` is replaced by a salted HMAC-SHA256, truncated to 16 hex chars. Salted,
    because the id space is small (an int): an unsalted hash of a patient id is trivially
    reversible by hashing every plausible integer. The salt comes from the environment,
    so traces from different deployments are not cross-linkable.
  * `patient_location` is coarsened. It is free text today, so it is truncated hard and
    kept only as a short token for aggregate drift analysis.
  * `medical_history` free text is **dropped by default** and reduced to a count. It is
    patient-reported prose and the most likely place for incidental identifiers ("my
    brother Dr Rao at Apollo said...").
  * Images are reduced to a SHA-256 of the bytes plus dimensions. Never the bytes, never
    a thumbnail, never a path that includes a patient directory.

The one honest limit: this protects what *we* put on spans. It cannot protect against a
future field being added straight into a span by code that does not route through here,
which is why `span_attributes()` is the only sanctioned way to put patient data on a span
and `test_redaction.py` asserts the name never appears in any emitted attribute.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

# Hard caps. Long strings on spans are a cost and a leak risk; free text is the worst
# offender, so it gets the tightest limit.
MAX_TEXT = 120
MAX_LOCATION = 32

_SALT_ENV = "SKINSURE_TRACE_SALT"

# Fields that must never reach a span, in any form. Checked by the tests as a
# belt-and-braces complement to the allowlist in `span_attributes()`.
NEVER_EMIT = frozenset({
    "patient_name",
    "image",
    "image_bytes",
    "raw_image",
    "metadata",          # the raw JSON blob /predict receives
})


def _salt() -> bytes:
    """HMAC salt for pseudonymising ids.

    Falls back to a fixed development value when unset so local runs work, but the
    fallback is deliberately obvious in the output (`pseudonymisation is NOT secure`
    shows up in `salt_source()`), because an unsalted-equivalent deployment would make
    `patient_id` hashes reversible across installs.
    """
    return os.environ.get(_SALT_ENV, "skinsure-dev-salt-not-for-production").encode()


def salt_source() -> str:
    """Whether a real salt is configured. Put this on the root span once, not the salt."""
    return "env" if os.environ.get(_SALT_ENV) else "dev-fallback (pseudonymisation is NOT secure)"


def pseudonymise(value: object) -> str:
    """Stable, salted, non-reversible id for correlating a patient's own requests.

    Same input + same salt -> same token, so a patient's requests can be grouped in
    traces without storing who they are. Different salt -> unlinkable, so traces from
    staging cannot be joined against production.
    """
    digest = hmac.new(_salt(), str(value).encode(), hashlib.sha256).hexdigest()
    return f"pid_{digest[:16]}"


def truncate(text: object, limit: int = MAX_TEXT) -> str:
    """Clip free text and mark that it was clipped, so a reader is not misled."""
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def coarsen_location(location: object | None) -> str | None:
    """Keep a short location token for drift analysis, not an address.

    `patient_location` is free text ("region, optional"), so there is no reliable way to
    cut it to state level here. Taking the first comma-separated component is the closest
    honest approximation: "Mumbai, Maharashtra" -> "Mumbai". It is then hard-truncated.
    If a structured location field is ever added, redact it properly instead of this.
    """
    if location is None:
        return None
    head = str(location).split(",")[0].strip()
    return truncate(head, MAX_LOCATION) or None


def image_fingerprint(data: bytes, *, width: int | None = None,
                      height: int | None = None) -> dict[str, Any]:
    """Identify an image without carrying any of it.

    SHA-256 of the exact bytes, so the same upload is recognisable across traces (useful
    for "the patient retried four times"), plus size and dimensions for the quality-gate
    and drift story. Never the bytes themselves and never a thumbnail.
    """
    out: dict[str, Any] = {
        "image.sha256": hashlib.sha256(data).hexdigest(),
        "image.bytes": len(data),
    }
    if width is not None and height is not None:
        out["image.width"] = int(width)
        out["image.height"] = int(height)
        out["image.short_side"] = int(min(width, height))
    return out


def span_attributes(patient: Any) -> dict[str, Any]:
    """The ONLY sanctioned way to put `PatientMetadata` on a span.

    An explicit allowlist: a field added to `PatientMetadata` later does not appear here
    until someone adds it deliberately. `patient_name` is absent by construction, not by
    filtering, so there is no ordering bug that could let it through.

    Takes `Any` rather than importing `PatientMetadata`, so this module stays dependency
    free and can redact a plain dict in a test just as well as the real model.
    """
    def get(field: str, default: Any = None) -> Any:
        if isinstance(patient, dict):
            return patient.get(field, default)
        return getattr(patient, field, default)

    def enum_value(field: str) -> Any:
        v = get(field)
        return getattr(v, "value", v)

    history = get("medical_history") or []
    attrs: dict[str, Any] = {
        # Pseudonymous, not identifying.
        "patient.pid": pseudonymise(get("patient_id")),
        # Clinically relevant and low-identifiability on their own.
        "patient.age": get("patient_age"),
        "patient.sex": enum_value("patient_sex"),
        "patient.fitzpatrick": enum_value("fitzpatrick_type"),
        "patient.body_area": enum_value("body_area"),
        "patient.symptom_duration_days": get("symptom_duration_days"),
        "patient.is_spreading_rapidly": bool(get("is_spreading_rapidly", False)),
        "patient.season": enum_value("current_season"),
        # Free text reduced to a count. The content stays out of traces entirely.
        "patient.medical_history_count": len(history),
    }
    loc = coarsen_location(get("patient_location"))
    if loc is not None:
        attrs["patient.location_coarse"] = loc
    return {k: v for k, v in attrs.items() if v is not None}


def redact(obj: Any) -> Any:
    """Best-effort recursive scrub for ad-hoc payloads (log extras, error context).

    `span_attributes()` is the right tool for patient metadata because it is an
    allowlist. This is the denylist fallback for arbitrary structures, where an allowlist
    is not possible — it drops `NEVER_EMIT` keys, pseudonymises anything that looks like
    an id, coarsens a location and truncates every string. Use it for debugging extras,
    never as the primary guard for known-sensitive data.
    """
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            k = str(key)
            if k in NEVER_EMIT:
                out[k] = "[redacted]"
            elif k == "patient_id":
                out["patient_pid"] = pseudonymise(value)
            elif k == "patient_location":
                out["patient_location_coarse"] = coarsen_location(value)
            elif k == "medical_history":
                out["medical_history_count"] = len(value or [])
            else:
                out[k] = redact(value)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return f"[{len(obj)} bytes, sha256={hashlib.sha256(bytes(obj)).hexdigest()[:16]}]"
    if isinstance(obj, str):
        return truncate(obj)
    return obj
