"""Tracing shim: fail-safe behaviour, attribute hygiene, and no content capture.

The tests that matter most here are the ones asserting tracing CANNOT break a request:
a None tracer, a missing OTel install and an exporter that raises must all leave the
instrumented code working.

Uses a local TracerProvider and `provider.get_tracer(...)` rather than
`trace.set_tracer_provider()`, because the global provider is once-per-process: a second
set is ignored with only a log warning, so a fixture that sets it would leak into every
later test and silently break the next swap.
"""
from __future__ import annotations

import pytest

from src.observability.tracing import (
    ERROR_TYPE,
    GEN_AI_CONTENT_ATTRS,
    GEN_AI_FINISH_REASONS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    SPAN_SCREENING,
    configure_tracing,
    get_tracer,
    llm_span_name,
    set_attrs,
    span,
)

pytest.importorskip("opentelemetry.sdk", reason="observability extra not installed")

from opentelemetry.sdk.trace import TracerProvider                      # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor          # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (    # noqa: E402
    InMemorySpanExporter,                 # NOT re-exported from .export; full path needed
)


@pytest.fixture
def traced():
    """A local provider + in-memory exporter. SimpleSpanProcessor, not Batch: batching is
    asynchronous and a test would race it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


# --- tracing must never break a request ----------------------------------------------

def test_span_with_none_tracer_is_a_noop():
    """Call sites are written unconditionally, so a None tracer must just work."""
    with span(None, "anything", {"k": "v"}) as sp:
        assert sp is None


def test_set_attrs_tolerates_none_span():
    set_attrs(None, {"k": "v"})          # must not raise


def test_set_attrs_never_raises_on_a_hostile_span():
    class Hostile:
        def set_attribute(self, *_args):
            raise RuntimeError("exporter exploded")

    set_attrs(Hostile(), {"k": "v"})     # swallowed: must not reach a clinical response


def test_configure_tracing_none_returns_no_provider():
    assert configure_tracing("none") is None


def test_configure_tracing_rejects_unknown_exporter():
    with pytest.raises(ValueError, match="unknown exporter"):
        configure_tracing("carrier-pigeon")


def test_get_tracer_without_provider_does_not_raise():
    tracer = get_tracer(None)
    with span(tracer, "s") as sp:
        if sp is not None:
            assert sp.is_recording() is False     # no-op span from the global API


def test_exception_is_recorded_then_reraised(traced):
    exporter, tracer = traced
    with pytest.raises(ValueError):
        with span(tracer, "boom"):
            raise ValueError("patient Zzyzx Qwertyuiop has a secret")

    (sp,) = exporter.get_finished_spans()
    assert sp.attributes[ERROR_TYPE] == "ValueError"
    # the message may carry patient data, so only the TYPE is recorded
    assert "Zzyzx" not in repr(sp.attributes)
    assert "secret" not in repr(sp.attributes)


# --- attributes reach the span, in OTel-safe form ------------------------------------

def test_span_records_attributes(traced):
    exporter, tracer = traced
    with span(tracer, SPAN_SCREENING, {"status": "completed", "escalated": True}):
        pass

    (sp,) = exporter.get_finished_spans()
    assert sp.name == SPAN_SCREENING
    assert sp.attributes["status"] == "completed"
    assert sp.attributes["escalated"] is True


def test_none_values_are_skipped_not_stringified(traced):
    exporter, tracer = traced
    with span(tracer, "s", {"present": 1, "absent": None}):
        pass

    (sp,) = exporter.get_finished_spans()
    assert "absent" not in sp.attributes
    assert "None" not in repr(sp.attributes)


def test_enums_are_reduced_to_values(traced):
    from api.schemas.patient import FitzpatrickType

    exporter, tracer = traced
    with span(tracer, "s", {"ftype": FitzpatrickType.V}):
        pass

    (sp,) = exporter.get_finished_spans()
    assert sp.attributes["ftype"] == "V"
    assert "FitzpatrickType" not in repr(sp.attributes)


def test_lists_become_homogeneous_string_sequences(traced):
    """`gen_ai.response.finish_reasons` is a string ARRAY, and triage reasons are a list."""
    exporter, tracer = traced
    with span(tracer, "s", {GEN_AI_FINISH_REASONS: ["end_turn"],
                            "triage.rules_fired": ["rule_2", "rule_4"]}):
        pass

    (sp,) = exporter.get_finished_spans()
    assert list(sp.attributes[GEN_AI_FINISH_REASONS]) == ["end_turn"]
    assert list(sp.attributes["triage.rules_fired"]) == ["rule_2", "rule_4"]


def test_empty_list_is_skipped(traced):
    exporter, tracer = traced
    with span(tracer, "s", {"rules": []}):
        pass
    (sp,) = exporter.get_finished_spans()
    assert "rules" not in sp.attributes


def test_bool_is_not_coerced_to_int(traced):
    """bool is an int subclass, so order of isinstance checks matters."""
    exporter, tracer = traced
    with span(tracer, "s", {"flag": False, "count": 0}):
        pass
    (sp,) = exporter.get_finished_spans()
    assert sp.attributes["flag"] is False
    assert sp.attributes["count"] == 0


# --- no prompt/completion content, ever ----------------------------------------------

def test_content_attributes_are_never_emitted(traced):
    """The Opt-In GenAI content attributes must not appear on any span we create.

    build_context() includes age, sex, Fitzpatrick type, body area, symptom duration and
    medical history. If content capture were ever enabled, that whole string would land
    on a span -- so assert the attributes are absent rather than trusting the default.
    """
    exporter, tracer = traced
    with span(tracer, llm_span_name("chat", "claude-haiku-5-5"),
              {GEN_AI_REQUEST_MODEL: "claude-haiku-5-5",
               GEN_AI_USAGE_INPUT_TOKENS: 1200}):
        pass

    (sp,) = exporter.get_finished_spans()
    for attr in GEN_AI_CONTENT_ATTRS:
        assert attr not in sp.attributes


def test_llm_span_name_follows_the_convention():
    assert llm_span_name("chat", "claude-haiku-5-5") == "chat claude-haiku-5-5"


# --- provider construction ------------------------------------------------------------

def test_console_exporter_builds_a_provider():
    provider = configure_tracing("console", service_name="skinsure-test")
    assert provider is not None
    assert provider.get_tracer("x") is not None


def test_configure_tracing_does_not_touch_the_global_by_default():
    """set_global defaults False because the global provider is once-per-process."""
    from opentelemetry import trace

    before = trace.get_tracer_provider()
    configure_tracing("console")
    assert trace.get_tracer_provider() is before


def test_nested_spans_form_one_trace(traced):
    exporter, tracer = traced
    with span(tracer, SPAN_SCREENING):
        with span(tracer, "ensemble"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child = next(s for s in spans if s.name == "ensemble")
    root = next(s for s in spans if s.name == SPAN_SCREENING)
    assert child.parent.span_id == root.context.span_id
    assert child.context.trace_id == root.context.trace_id
