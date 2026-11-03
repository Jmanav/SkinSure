"""`run_screening` decision spans: shape, parentage, and that nothing leaks.

Two properties carry most of the value here:

* **The triage span is the safety audit.** `triage.rules_fired` plus
  `severity_before`/`severity_after` are what let someone answer "which rule escalated
  this case, and from what?" months later. The escalation test below is the one that
  would notice if that evidence stopped being recorded.
* **Instrumentation is observation only.** `test_tracer_none_matches_traced_result`
  compares the ScreeningResult with and without a tracer field by field, so a span that
  accidentally changed control flow or a return value fails rather than passing quietly.

Local TracerProvider + `provider.get_tracer(...)`, never `trace.set_tracer_provider()`:
the global provider is once-per-process, so a fixture that set it would leak into every
later test and silently defeat the next swap.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

import numpy as np
import pytest
from helpers import FakeClient, llm_json
from PIL import Image

from api.schemas.predictions import DiseaseClass, ScreeningStatus, Severity
from src.agents.orchestrator import run_screening
from src.observability.tracing import (
    GEN_AI_CONTENT_ATTRS,
    SPAN_ENSEMBLE,
    SPAN_EXPLAINER,
    SPAN_QUALITY_GATE,
    SPAN_SCREENING,
    SPAN_SUMMARIZER,
    SPAN_TRIAGE,
)

pytest.importorskip("opentelemetry.sdk", reason="observability extra not installed")

from opentelemetry.sdk.trace import TracerProvider                      # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor          # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (    # noqa: E402
    InMemorySpanExporter,                 # NOT re-exported from .export; full path needed
)

# A name no clinical string could contain by chance, so finding it anywhere in a span
# attribute is proof of a leak rather than a coincidence.
LEAKY_NAME = "Zzyzx Qwertyuiop"


@pytest.fixture
def traced():
    """Local provider + in-memory exporter. SimpleSpanProcessor, not Batch: batching is
    asynchronous and a test would race it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


class FakeEnsemble:
    def __init__(self, predictions):
        self._predictions = predictions
        self.calls = 0

    def predict(self, image_tensor):
        self.calls += 1
        return self._predictions


class FakeExplainer:
    def __init__(self, xai):
        self._xai = xai
        self.calls = 0

    def __call__(self, image_tensor, predictions):
        self.calls += 1
        return self._xai


def _noise_image(size: int = 256) -> Image.Image:
    """Random noise: big enough and sharp enough to pass the quality gate."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _screen(image, patient, predictions, xai, *, tracer=None, llm_text=None):
    """Run a screening against fakes only -- no checkpoints, no network."""
    return asyncio.run(
        run_screening(
            image,
            patient,
            ensemble=FakeEnsemble(predictions),
            explainer=FakeExplainer(xai),
            client=FakeClient(llm_text or llm_json()),
            tracer=tracer,
        )
    )


def _by_name(exporter) -> dict[str, object]:
    return {sp.name: sp for sp in exporter.get_finished_spans()}


# --- all three exit paths are attributed ----------------------------------------------

def test_completed_path_attributes_the_root_span(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    result = _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    assert result.status is ScreeningStatus.COMPLETED
    root = _by_name(exporter)[SPAN_SCREENING]
    assert root.attributes["screening.status"] == "completed"
    assert root.attributes["screening.escalated"] is False
    assert root.attributes["screening.treatment_advice_allowed"] is True
    assert root.attributes["screening.final_severity"] == "routine"


def test_rejected_path_attributes_the_root_span(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    result = _screen(
        _noise_image(size=64), make_patient(), make_predictions(), make_xai(), tracer=tracer
    )

    assert result.status is ScreeningStatus.REJECTED
    root = _by_name(exporter)[SPAN_SCREENING]
    assert root.attributes["screening.status"] == "rejected"
    assert root.attributes["screening.treatment_advice_allowed"] is False


def test_blocked_path_attributes_the_root_span(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    preds = make_predictions(primary_confidence=0.30, secondary_confidence=0.20)
    result = _screen(_noise_image(), make_patient(), preds, make_xai(), tracer=tracer)

    assert result.status is ScreeningStatus.BLOCKED
    root = _by_name(exporter)[SPAN_SCREENING]
    assert root.attributes["screening.status"] == "blocked"
    assert root.attributes["screening.treatment_advice_allowed"] is False
    # A blocked result withholds the summary, so no final severity is published for it.
    assert "screening.final_severity" not in root.attributes


def test_rejected_path_has_a_quality_gate_span_and_no_ensemble_span(
    traced, make_patient, make_predictions, make_xai
):
    """The gate is the decision that ended the request, so it must be recorded -- and the
    absence of an ensemble span is the evidence the model never ran."""
    exporter, tracer = traced
    _screen(_noise_image(size=64), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    spans = _by_name(exporter)
    gate = spans[SPAN_QUALITY_GATE]
    assert gate.attributes["quality.ok"] is False
    assert gate.attributes["quality.decision"] == "reject"
    assert gate.attributes["quality.reason_count"] >= 1
    assert SPAN_ENSEMBLE not in spans
    assert SPAN_SUMMARIZER not in spans
    assert SPAN_TRIAGE not in spans


# --- the step spans -------------------------------------------------------------------

def test_quality_gate_records_a_pass(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    gate = _by_name(exporter)[SPAN_QUALITY_GATE]
    assert gate.attributes["quality.ok"] is True
    assert gate.attributes["quality.decision"] == "pass"
    assert gate.attributes["quality.reason_count"] == 0


def test_quality_gate_reason_text_never_reaches_the_span(
    traced, make_patient, make_predictions, make_xai
):
    """quality.reasons are patient-readable strings; only counts belong on a span."""
    exporter, tracer = traced
    _screen(_noise_image(size=64), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    gate = _by_name(exporter)[SPAN_QUALITY_GATE]
    assert "quality.reasons" not in gate.attributes
    assert "too small" not in repr(gate.attributes)


def test_ensemble_span_carries_the_prediction_and_tier(
    traced, make_patient, make_predictions, make_xai
):
    exporter, tracer = traced
    preds = make_predictions(agreement_score=0.67)
    _screen(_noise_image(), make_patient(), preds, make_xai(), tracer=tracer)

    sp = _by_name(exporter)[SPAN_ENSEMBLE]
    assert sp.attributes["ensemble.primary_label"] == "Eczema"
    assert sp.attributes["ensemble.primary_confidence"] == pytest.approx(0.80)
    assert sp.attributes["ensemble.secondary_label"] == "Keratosis"
    # Taken from Predictions.confidence_tier(), not re-derived here either.
    assert sp.attributes["ensemble.confidence_tier"] == preds.confidence_tier() == "moderate"


def test_explainer_span_carries_the_xai_summary(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    _screen(
        _noise_image(), make_patient(), make_predictions(),
        make_xai(artifact_flag=True), tracer=tracer,
    )

    sp = _by_name(exporter)[SPAN_EXPLAINER]
    assert sp.attributes["xai.focus_description"] == "lesion_centre"
    assert sp.attributes["xai.boundary_alignment"] == "good"
    assert sp.attributes["xai.artifact_flag"] is True


def test_summarizer_span_has_decisions_but_no_content(
    traced, make_patient, make_predictions, make_xai
):
    """The LLM's verdict fields only -- never the prose it wrote or the context it read."""
    exporter, tracer = traced
    _screen(
        _noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer,
        llm_text=llm_json(severity="monitor", summary="A distinctive lesion on the arm."),
    )

    sp = _by_name(exporter)[SPAN_SUMMARIZER]
    assert sp.attributes["summarizer.severity_suggested"] == "monitor"
    assert sp.attributes["summarizer.confidence_level"] == "high"
    assert sp.attributes["summarizer.recapture_needed"] is False
    assert sp.attributes["summarizer.immediate_referral"] is False
    assert "distinctive lesion" not in repr(sp.attributes)
    assert "Moisturise" not in repr(sp.attributes)


# --- the triage span: the safety audit ------------------------------------------------

def test_triage_span_records_the_escalation(traced, make_patient, make_predictions, make_xai):
    """Melanoma at 0.70 is well over MELANOMA_URGENT_MIN_CONFIDENCE (0.40), so Rule 2
    fires against an LLM that proposed `routine`."""
    exporter, tracer = traced
    preds = make_predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.70,
        secondary_label=DiseaseClass.KERATOSIS,
    )
    result = _screen(
        _noise_image(), make_patient(), preds, make_xai(), tracer=tracer,
        llm_text=llm_json(severity="routine"),
    )

    assert result.summary.severity is Severity.URGENT and result.escalated is True

    sp = _by_name(exporter)[SPAN_TRIAGE]
    fired = list(sp.attributes["triage.rules_fired"])
    assert fired and any("melanoma at confidence" in r for r in fired)
    assert sp.attributes["triage.severity_before"] == "routine"
    assert sp.attributes["triage.severity_after"] == "urgent"
    assert sp.attributes["triage.escalated"] is True
    assert sp.attributes["triage.blocked"] is False
    # The root span must agree with the triage span about the outcome.
    root = _by_name(exporter)[SPAN_SCREENING]
    assert root.attributes["screening.escalated"] is True
    assert root.attributes["screening.final_severity"] == "urgent"


def test_triage_span_records_a_block(traced, make_patient, make_predictions, make_xai):
    exporter, tracer = traced
    preds = make_predictions(primary_confidence=0.30, secondary_confidence=0.20)
    _screen(_noise_image(), make_patient(), preds, make_xai(), tracer=tracer)

    sp = _by_name(exporter)[SPAN_TRIAGE]
    assert sp.attributes["triage.blocked"] is True
    assert any("primary_confidence" in r for r in sp.attributes["triage.rules_fired"])


def test_triage_span_omits_rules_fired_when_nothing_fired(
    traced, make_patient, make_predictions, make_xai
):
    """An empty list is skipped by the shim, so "no rules fired" reads as an absent
    attribute rather than an empty array -- before/after still prove no downgrade."""
    exporter, tracer = traced
    _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    sp = _by_name(exporter)[SPAN_TRIAGE]
    assert "triage.rules_fired" not in sp.attributes
    assert sp.attributes["triage.severity_before"] == sp.attributes["triage.severity_after"]


# --- trace structure ------------------------------------------------------------------

def test_step_spans_are_children_of_the_screening_root(
    traced, make_patient, make_predictions, make_xai
):
    exporter, tracer = traced
    _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    spans = _by_name(exporter)
    root = spans[SPAN_SCREENING]
    for name in (SPAN_QUALITY_GATE, SPAN_ENSEMBLE, SPAN_EXPLAINER, SPAN_SUMMARIZER, SPAN_TRIAGE):
        child = spans[name]
        assert child.parent is not None, name
        assert child.parent.span_id == root.context.span_id, name
        assert child.context.trace_id == root.context.trace_id, name


# --- nothing leaks --------------------------------------------------------------------

@pytest.mark.parametrize(
    "image_size, preds_kwargs",
    [
        (256, {}),                                              # completed
        (64, {}),                                               # rejected
        (256, {"primary_confidence": 0.30, "secondary_confidence": 0.20}),   # blocked
    ],
    ids=["completed", "rejected", "blocked"],
)
def test_patient_name_never_appears_on_any_span(
    traced, make_patient, make_predictions, make_xai, image_size, preds_kwargs
):
    """span_attributes() drops patient_name by construction; this asserts it across every
    exit path, since a leak on the rarely-exercised branch is the one that survives."""
    exporter, tracer = traced
    patient = make_patient(patient_name=LEAKY_NAME)
    _screen(
        _noise_image(size=image_size), patient,
        make_predictions(**preds_kwargs), make_xai(), tracer=tracer,
    )

    spans = exporter.get_finished_spans()
    assert spans
    for sp in spans:
        assert "Zzyzx" not in repr(sp.attributes), sp.name
        assert "Qwertyuiop" not in repr(sp.attributes), sp.name


def test_root_span_carries_redacted_patient_context(
    traced, make_patient, make_predictions, make_xai
):
    exporter, tracer = traced
    patient = make_patient(patient_name=LEAKY_NAME, patient_id=4242)
    _screen(_noise_image(), patient, make_predictions(), make_xai(), tracer=tracer)

    root = _by_name(exporter)[SPAN_SCREENING]
    assert root.attributes["patient.fitzpatrick"] == "V"
    assert root.attributes["patient.age"] == 45
    assert root.attributes["patient.pid"].startswith("pid_")
    assert "4242" not in root.attributes["patient.pid"]


def test_no_gen_ai_content_attributes_on_any_span(
    traced, make_patient, make_predictions, make_xai
):
    exporter, tracer = traced
    _screen(_noise_image(), make_patient(), make_predictions(), make_xai(), tracer=tracer)

    for sp in exporter.get_finished_spans():
        for attr in GEN_AI_CONTENT_ATTRS:
            assert attr not in sp.attributes, f"{attr} on {sp.name}"


# --- instrumentation is observation only ----------------------------------------------

@pytest.mark.parametrize(
    "image_size, preds_kwargs",
    [
        (256, {}),
        (64, {}),
        (256, {"primary_confidence": 0.30, "secondary_confidence": 0.20}),
    ],
    ids=["completed", "rejected", "blocked"],
)
def test_tracer_none_matches_traced_result(
    traced, make_patient, make_predictions, make_xai, image_size, preds_kwargs
):
    """The default (tracer=None) must be byte-identical to a traced run on every path."""
    _, tracer = traced
    args = (_noise_image(size=image_size), make_patient())
    kwargs = dict(predictions=make_predictions(**preds_kwargs), xai=make_xai())

    untraced = _screen(*args, kwargs["predictions"], kwargs["xai"])
    instrumented = _screen(*args, kwargs["predictions"], kwargs["xai"], tracer=tracer)

    assert asdict(untraced) == asdict(instrumented)


def test_default_signature_still_works_without_a_tracer_argument(
    make_patient, make_predictions, make_xai
):
    """Existing callers (api/main.py, tests/unit/test_orchestrator.py) pass no tracer."""
    result = asyncio.run(
        run_screening(
            _noise_image(),
            make_patient(),
            ensemble=FakeEnsemble(make_predictions()),
            explainer=FakeExplainer(make_xai()),
            client=FakeClient(llm_json()),
        )
    )
    assert result.status is ScreeningStatus.COMPLETED


def test_an_exception_mid_pipeline_is_recorded_and_reraised(
    traced, make_patient, make_predictions, make_xai
):
    """Tracing must not swallow a real failure: a bad LLM response still raises, and the
    spans that were open record the error type (never the message)."""
    exporter, tracer = traced
    with pytest.raises(ValueError):
        _screen(
            _noise_image(), make_patient(), make_predictions(), make_xai(),
            tracer=tracer, llm_text="not json",
        )

    spans = _by_name(exporter)
    assert spans[SPAN_SUMMARIZER].attributes["error.type"] == "ValueError"
    assert spans[SPAN_SCREENING].attributes["error.type"] == "ValueError"
    assert SPAN_TRIAGE not in spans
