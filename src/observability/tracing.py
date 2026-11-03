"""Tracing setup and the span-attribute vocabulary.

Design decisions, and why they differ from the written plan:

**Tracers are injected, never fetched from the global provider at module scope.**
`trace.set_tracer_provider()` is once-per-process: a second call is ignored with only a
log warning, so a test that sets a global provider leaks into every later test and the
next swap silently does nothing. (This is a real bug class -- it bit pydantic/logfire.)
`get_tracer()` here resolves lazily *per call*, and `configure_tracing()` returns the
provider so a test can hold its own and never touch the global.

**No `@traced` decorator.** `run_screening` is five sequential steps with no seams, so a
decorator buys nothing a context manager doesn't. What matters is which *decisions* get
recorded, not which functions ran -- `triage`'s `rules_fired` is worth more than ten
timing spans, because it is what an auditor or a clinician will ask about.

**All `gen_ai.*` names live in this module only.** Every one of them is at semconv
stability level *Development*, the lowest tier: they may be renamed or removed in any
minor release with no deprecation period. The GenAI conventions also now live in their
own repository (`open-telemetry/semantic-conventions-genai`, split out at semconv
v1.42.0) and version independently of core semconv, so churn is likely. Centralising
them means a rename is one edit here, not a scatter-gun change across call sites.

**Content capture is off, and stays off.** Prompts and completions would travel in
`gen_ai.input.messages` / `gen_ai.output.messages`, which are *Opt-In* attributes. We
never set them, and we never set the env vars that would make conforming instrumentation
set them (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, which defaults to
`NO_CONTENT`, and `OTEL_SEMCONV_STABILITY_OPT_IN`). This matters because
`build_context()` includes age, sex, Fitzpatrick type, body area, symptom duration and
history -- if content capture were ever switched on, that whole string would land on a
span. A test asserts those attributes are absent.

**Tracing cannot break a request.** That is a spec MUST, not a courtesy: OTel
implementations "MUST NOT throw unhandled exceptions at runtime", with no provider the
API degrades to a no-op returning non-recording spans, and export failures are logged and
dropped on a background thread. Verified locally: with no provider configured,
`start_as_current_span` does not raise and `is_recording()` is False. The guarantee
covers OTel's own operations, NOT our attribute-building code -- so `set_attrs()` here
swallows its own errors, because a `None` reaching `set_attribute` must never surface in
a clinical response.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# --- span names ----------------------------------------------------------------------
SPAN_SCREENING = "screening"           # root: one per /predict call, the unit of evals
SPAN_QUALITY_GATE = "quality_gate"
SPAN_ENSEMBLE = "ensemble"
SPAN_EXPLAINER = "explainer.gradcam"
SPAN_SUMMARIZER = "summarizer"
SPAN_TRIAGE = "triage"

# --- GenAI semantic conventions (all stability: Development -- see module docstring) ---
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens"
GEN_AI_FINISH_REASONS = "gen_ai.response.finish_reasons"   # string[], plural
ERROR_TYPE = "error.type"                                   # this one IS Stable

# Opt-In content attributes. Listed so the test can assert we never emit them.
GEN_AI_CONTENT_ATTRS = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
})

#: Span name for an LLM inference span: "{operation} {model}".
def llm_span_name(operation: str, model: str) -> str:
    return f"{operation} {model}"


_PRIMITIVES = (str, bool, int, float)


def _safe_value(value: Any) -> Any:
    """Coerce to something OTel will accept, or return None to skip the attribute.

    OTel takes str/bool/int/float and homogeneous sequences of those; anything else is
    dropped or stringified unpredictably. Enums are reduced to `.value`. Note bool must
    be checked before int, since bool is an int subclass.
    """
    if value is None:
        return None
    value = getattr(value, "value", value)      # enums
    if isinstance(value, bool):
        return value
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, (list, tuple)):
        items = [v for v in (_safe_value(v) for v in value) if v is not None]
        return [str(v) for v in items] if items else None
    return str(value)


def set_attrs(span: Any, attrs: dict[str, Any]) -> None:
    """Set attributes on a span, skipping None and never raising.

    OTel's no-throw guarantee covers its own operations, not ours. A bad attribute value
    must not turn into a 500 on a clinical endpoint, so failures here are logged at debug
    and dropped.
    """
    if span is None:
        return
    for key, raw in attrs.items():
        value = _safe_value(raw)
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:                        # pragma: no cover - defensive
            logger.debug("could not set span attribute %s", key, exc_info=True)


def configure_tracing(
    exporter: str = "none",
    *,
    service_name: str = "skinsure",
    set_global: bool = False,
) -> Any:
    """Build a TracerProvider for `exporter`, or return None when tracing is off.

    `exporter` is one of "none" | "console" | "otlp". Returns the provider so the caller
    owns it; `set_global` is opt-in and defaults to False precisely because setting the
    global provider is irreversible within a process.

    Import of OTel is deferred and failure is non-fatal: the app must boot and serve
    without a tracing stack installed. That is also why `observability` is an optional
    dependency group rather than a core one -- a clinical API should not require a
    telemetry library to start.
    """
    if exporter == "none":
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.warning(
            "tracing requested (exporter=%s) but opentelemetry-sdk is not installed; "
            "continuing without tracing. Install with: pip install -e .[observability]",
            exporter,
        )
        return None

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))

    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            logger.warning("exporter=otlp but opentelemetry-exporter-otlp-proto-http is "
                           "not installed; continuing without an exporter")
            return provider
        # Endpoint and headers come from the standard OTEL_EXPORTER_OTLP_* env vars, so
        # no backend URL or credential is ever written into this repo. For Langfuse, note
        # the x-langfuse-ingestion-version=4 header: without it, directly exported spans
        # can take up to 10 minutes to appear, which reads exactly like "tracing is
        # broken". Langfuse accepts OTLP/HTTP protobuf and does NOT support gRPC, which
        # is why this is the proto-http exporter.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        raise ValueError(f"unknown exporter {exporter!r}; expected none|console|otlp")

    if set_global:
        from opentelemetry import trace
        trace.set_tracer_provider(provider)
    return provider


def get_tracer(provider: Any = None, name: str = "skinsure") -> Any:
    """A tracer from `provider`, or a no-op tracer when tracing is unavailable.

    Resolved lazily per call rather than captured at import time, so a provider created
    after import is still picked up and tests can pass their own. With OTel absent this
    returns None and every call site treats that as "do not trace".
    """
    if provider is not None:
        return provider.get_tracer(name)
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    return trace.get_tracer(name)


@contextmanager
def span(tracer: Any, name: str, attrs: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start a span, or do nothing at all when `tracer` is None.

    The None branch is what lets instrumented code be written unconditionally: call
    sites never test whether tracing is on. Exceptions propagate (the request should
    still fail if the work failed) but are recorded first, with the exception *type*
    only -- never a message, which could contain patient data from a validation error.
    """
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as sp:
        if attrs:
            set_attrs(sp, attrs)
        try:
            yield sp
        except Exception as exc:
            set_attrs(sp, {ERROR_TYPE: type(exc).__qualname__})
            raise
