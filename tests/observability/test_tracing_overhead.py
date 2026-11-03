"""The two A1 gate tests: span overhead, and an exporter outage not breaking a request.

**On the overhead test: it is a SMOKE TEST, not the measurement.**

The real A1 gate is "tracing costs <5% of a /predict request". That ratio cannot be
measured here and does not need to be: /predict is *seconds* long - Grad-CAM alone is
2.6-6.3 s mean and up to 15.7 s on the dev CPU (CLAUDE.md gap 6), before the ensemble
and an LLM round trip - while a span costs *microseconds*. The gate therefore passes by
roughly three orders of magnitude, and arithmetic settles it more honestly than a timing
loop on a busy box would. What this test exists for is to catch a CATASTROPHIC
regression: somebody adding a synchronous network export, a blocking flush on span end,
or an O(n) attribute copy, which would move the per-span cost from microseconds to
milliseconds and show up here as a huge ratio.

So the assertions are deliberately loose and the numbers are REPORTED rather than
asserted tightly. This box is a 1-physical-core machine under load; a tight bound would
flake, and a flaky gate test gets deleted or `-k`'d out, which is worse than a loose one.
Design choices that follow from that:

  * the timed unit of work is a real ~1 ms numpy computation, not a no-op. Timing `pass`
    with and without a span measures the span and nothing else, so any ratio is
    meaningless as a percentage - the denominator has to be work of the kind being
    instrumented. It is four passes of a cheap vectorised op rather than one big
    LAPACK call on purpose: an SVD of the same duration has a 20-30 ms spread on this
    box, which swamps a microsecond signal and can even make the traced median come
    out *faster* than the baseline.
  * MEDIANS of per-iteration samples, not means of block totals: one GC pause or
    scheduler preemption ruins a mean and leaves a median untouched.
  * OFF and ON are interleaved sample by sample, so a slow stretch of the machine's
    load cannot land entirely on one arm of the comparison.
  * time.perf_counter, the highest-resolution monotonic clock available.
  * if the OFF baseline is itself wildly unstable, the box is too noisy for any ratio to
    mean anything, and the test skips rather than failing spuriously.
  * the measured number is PESSIMISTIC: SimpleSpanProcessor exports synchronously on
    span end, so the serialise-and-hand-off cost lands inside the timed region. The
    production path (configure_tracing) uses BatchSpanProcessor, which only enqueues.
    Measured here at ~250 us/span; that is the ceiling, not the shipped cost.

The exporter-outage test is the one that genuinely matters, and it is strict.
"""
from __future__ import annotations

import statistics
import time

import pytest

from src.observability.tracing import SPAN_SCREENING, configure_tracing, span

pytest.importorskip("opentelemetry.sdk", reason="observability extra not installed")

import numpy as np                                                      # noqa: E402
from opentelemetry.sdk.trace import TracerProvider                      # noqa: E402
from opentelemetry.sdk.trace.export import (                            # noqa: E402
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (    # noqa: E402
    InMemorySpanExporter,
)

# A ~1 ms unit of work, so the span cost is a small fraction of a real denominator
# rather than the whole measurement. 4 x sqrt+sum over 60k float64 measures ~1.1 ms
# median on this box with an interquartile spread well under 0.1 ms.
_WORK = np.random.default_rng(0).random(60_000)
_WORK_PASSES = 4

SAMPLES = 150        # per-iteration timings per arm, interleaved
# Generous: the honest bound is ~0% of a seconds-long request. 50% of a ~1 ms unit of
# work means a per-span cost under ~500 us, which still clears the real gate by orders
# of magnitude while catching a regression into milliseconds.
MAX_OVERHEAD_RATIO = 0.50


def _work() -> float:
    """~1 ms of real computation. Returns a value so it cannot be optimised away."""
    total = 0.0
    for _ in range(_WORK_PASSES):
        total += float(np.sqrt(_WORK).sum())
    return total


def _timed(tracer) -> float:
    """Seconds for one unit of work, with a span around it when tracer is not None."""
    start = time.perf_counter()
    with span(tracer, SPAN_SCREENING, {"work": 1}):
        _work()
    return time.perf_counter() - start


def test_span_overhead_is_small_relative_to_real_work():
    """SMOKE TEST (see module docstring): catches a regression into milliseconds."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("overhead")

    for _ in range(20):                       # warm up numpy/BLAS and the tracing path
        _timed(None)
        _timed(tracer)
    exporter.clear()

    off_samples, on_samples = [], []
    for _ in range(SAMPLES):                  # interleaved, not two separate blocks
        off_samples.append(_timed(None))
        on_samples.append(_timed(tracer))

    off = statistics.median(off_samples)
    on = statistics.median(on_samples)
    per_span_us = (on - off) * 1e6
    ratio = (on - off) / off

    print(
        f"\n[overhead] work unit {off * 1e3:.3f} ms  traced {on * 1e3:.3f} ms  "
        f"per-span cost {per_span_us:.1f} us  ratio {ratio * 100:+.1f}%  "
        f"(n={SAMPLES} per arm; bound {MAX_OVERHEAD_RATIO * 100:.0f}% of a ~1 ms unit. "
        f"The real A1 gate is <5% of a seconds-long /predict, which a microsecond span "
        f"clears by ~1000x; this is a regression smoke test, not that measurement.)"
    )

    # Spans really were produced - otherwise this would time an accidental no-op.
    assert len(exporter.get_finished_spans()) == SAMPLES

    # If the baseline is unstable the ratio is noise, not a measurement. Compared on the
    # interquartile range, since a single outlier sample is expected on a loaded box.
    quartiles = statistics.quantiles(off_samples, n=4)
    iqr_fraction = (quartiles[2] - quartiles[0]) / off
    if iqr_fraction > 0.40:
        pytest.skip(
            f"timing too unstable to compare (baseline IQR {iqr_fraction:.0%} of median)"
        )

    assert ratio < MAX_OVERHEAD_RATIO, (
        f"span overhead {ratio:.1%} of a ~{off * 1e3:.2f} ms unit of work "
        f"({per_span_us:.0f} us/span) - expected well under "
        f"{MAX_OVERHEAD_RATIO:.0%}; suspect a synchronous export or blocking flush"
    )


# --- exporter outage must not break a request -----------------------------------------

class RaisingExporter(SpanExporter):
    """An exporter whose export() raises - the backend being down, in effect."""

    def __init__(self) -> None:
        self.attempts = 0

    def export(self, spans):                      # noqa: D102
        self.attempts += 1
        raise RuntimeError("exporter is down")

    def shutdown(self) -> None:                   # noqa: D102
        pass


class SlowExporter(SpanExporter):
    """An exporter that stalls, i.e. a hung backend rather than a failing one."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.attempts = 0

    def export(self, spans):                      # noqa: D102
        self.attempts += 1
        time.sleep(self.delay)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:                   # noqa: D102
        pass


def _tracer_with(exporter) -> object:
    """SimpleSpanProcessor, NOT Batch.

    Batch exports on a background thread, which would swallow the failure and leave the
    test proving nothing. Simple exports synchronously on span end, so if an exporter
    exception could escape into the request path, it escapes here.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("outage")


def test_raising_exporter_does_not_break_the_wrapped_work():
    exporter = RaisingExporter()
    tracer = _tracer_with(exporter)

    def screen() -> str:
        with span(tracer, SPAN_SCREENING, {"patient.fitzpatrick": "V"}):
            return "completed"

    assert screen() == "completed"
    assert exporter.attempts > 0, "exporter was never called; the test proved nothing"


def test_raising_exporter_does_not_propagate_out_of_the_span_block():
    """The export failure must not surface at the `with span(...)` boundary."""
    exporter = RaisingExporter()
    tracer = _tracer_with(exporter)

    reached_the_end = False
    with span(tracer, SPAN_SCREENING) as sp:
        assert sp is not None
        reached_the_end = True
    # No pytest.raises here on purpose: the point is that nothing is raised at all,
    # including on __exit__ where the synchronous export happens.
    assert reached_the_end
    assert exporter.attempts > 0


def test_many_consecutive_export_failures_still_complete():
    """A sustained outage, not a single blip - no cumulative failure mode."""
    exporter = RaisingExporter()
    tracer = _tracer_with(exporter)
    results = []
    for i in range(25):
        with span(tracer, SPAN_SCREENING, {"i": i}):
            results.append(i)
    assert results == list(range(25))
    assert exporter.attempts == 25


def test_stalled_exporter_still_returns_the_value():
    """A hung backend delays, but must not fail or lose the request's result."""
    exporter = SlowExporter(delay=0.02)
    tracer = _tracer_with(exporter)
    with span(tracer, SPAN_SCREENING):
        value = {"status": "completed"}
    assert value == {"status": "completed"}
    assert exporter.attempts == 1


def test_work_exception_still_propagates_through_a_failing_exporter():
    """Fail-safe is not fail-silent: a real failure in the work must still surface,
    even when the exporter is also broken and recording the error itself fails."""
    tracer = _tracer_with(RaisingExporter())
    with pytest.raises(ValueError, match="quality gate"):
        with span(tracer, SPAN_SCREENING):
            raise ValueError("quality gate rejected the image")


def test_configure_otlp_without_an_endpoint_does_not_raise(monkeypatch):
    """Nothing configured, so no endpoint: construction must still succeed.

    pytest-socket blocks egress (--disable-socket --allow-hosts=127.0.0.1), and that is
    fine - no export is attempted here. What is asserted is that building the exporter
    and provider is not where an unconfigured deployment falls over. An actual outage is
    simulated above with RaisingExporter rather than by touching the network.
    """
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        monkeypatch.delenv(var, raising=False)

    provider = configure_tracing("otlp", service_name="skinsure-test")
    # None is a legitimate outcome (exporter package absent); a provider is the other.
    if provider is not None:
        tracer = provider.get_tracer("otlp-no-endpoint")
        with span(tracer, SPAN_SCREENING, {"k": "v"}) as sp:
            assert sp is not None
        provider.shutdown()      # must not raise either, even with nowhere to send to
