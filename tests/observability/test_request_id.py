"""Request-id correlation, including the inbound header as untrusted input.

Two things are being tested, and only one of them is plumbing:

  * correlation -- an id reaches every response, including the failures, because a
    complaint is almost always about a failure and the id is the only handle on it.
    That includes the unhandled 500, which needs a second mechanism (see the 500 tests):
    middleware alone provably cannot reach it.
  * the injection boundary -- `X-Request-ID` is attacker-controlled and lands in both a
    response header and a log line, so the tests that matter are the ones asserting a
    hostile value is *replaced*, not filtered. Asserting "no CRLF in the header" would
    pass against a sanitiser that still echoed the attacker's prefix; these assert the
    submitted substring is absent from the headers entirely.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from helpers import FakeClient, llm_json
from PIL import Image

from api.main import MAX_REQUEST_ID_LEN, REQUEST_ID_HEADER, create_app

PATIENT = dict(
    patient_id=1,
    patient_name="Asha Verma",
    patient_age=45,
    patient_sex="female",
    fitzpatrick_type="V",
    body_area="arm",
    symptom_duration_days=30,
)


class FakeEnsemble:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, image_tensor):
        return self.predictions


def _png_bytes(size: int = 256) -> bytes:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_client(make_predictions, make_xai):
    """Same injection pattern as tests/integration: no checkpoints, no network."""
    def _make(explainer="default"):
        if explainer == "default":
            xai = make_xai()
            explainer = lambda tensor, preds: xai  # noqa: E731
        app = create_app(
            ensemble=FakeEnsemble(make_predictions()),
            explainer=explainer,
            client=FakeClient(llm_json()),
        )
        return TestClient(app)

    return _make


def _post(client, **kwargs):
    files = {"image": ("lesion.png", _png_bytes(), "image/png")}
    data = {"metadata": json.dumps(PATIENT)}
    with client as c:
        return c.post("/predict", files=files, data=data, **kwargs)


# --- the id is always present ---------------------------------------------------------

def test_health_response_carries_a_request_id(make_client):
    with make_client() as c:
        r = c.get("/health")

    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER]


def test_generated_ids_differ_between_requests(make_client):
    with make_client() as c:
        first = c.get("/health").headers[REQUEST_ID_HEADER]
        second = c.get("/health").headers[REQUEST_ID_HEADER]

    assert first and second and first != second


def test_predict_success_carries_a_request_id(make_client):
    r = _post(make_client())

    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER]


# --- a valid caller-supplied id is reused ---------------------------------------------

def test_supplied_id_is_echoed_unchanged(make_client):
    supplied = "gateway-abc_123DEF"
    with make_client() as c:
        r = c.get("/health", headers={REQUEST_ID_HEADER: supplied})

    assert r.headers[REQUEST_ID_HEADER] == supplied


def test_supplied_id_at_the_length_limit_is_still_reused(make_client):
    """The cap is a boundary, so the boundary itself is pinned -- 128 is accepted."""
    supplied = "a" * MAX_REQUEST_ID_LEN
    with make_client() as c:
        r = c.get("/health", headers={REQUEST_ID_HEADER: supplied})

    assert r.headers[REQUEST_ID_HEADER] == supplied


# --- hostile values are replaced, not sanitised ---------------------------------------

def test_oversized_id_is_replaced(make_client):
    oversized = "x" * 500
    with make_client() as c:
        r = c.get("/health", headers={REQUEST_ID_HEADER: oversized})

    returned = r.headers[REQUEST_ID_HEADER]
    assert returned != oversized
    assert len(returned) <= MAX_REQUEST_ID_LEN
    # not merely truncated: a truncated prefix is still attacker-chosen content
    assert oversized[:32] not in returned


@pytest.mark.parametrize(
    "hostile, marker",
    [
        ("abc\r\nInjected: evil", "Injected"),      # header splitting
        ("abc\x00def", "def"),                      # NUL
        ("abc\ndef", "def"),                        # bare LF -- log forging
        ("abc def", "def"),                         # space: not a token char
        ("abc;path=/", "path"),                     # punctuation outside the allowlist
        ("../../etc/passwd", "passwd"),             # traversal, in case an id names a file
    ],
)
def test_unsafe_id_is_replaced_and_its_content_never_reaches_the_headers(
    make_client, hostile, marker
):
    with make_client() as c:
        r = c.get("/health", headers={REQUEST_ID_HEADER: hostile})

    returned = r.headers[REQUEST_ID_HEADER]
    assert returned != hostile
    assert marker not in returned
    # nothing the caller submitted may appear anywhere in the response headers, under
    # any header name -- not just under X-Request-ID
    assert marker not in "\n".join(f"{k}: {v}" for k, v in r.headers.items())


def test_injected_header_is_not_present_as_its_own_header(make_client):
    """If splitting had worked, `Injected` would exist as a real response header."""
    with make_client() as c:
        r = c.get("/health", headers={REQUEST_ID_HEADER: "abc\r\nInjected: evil"})

    assert "Injected" not in r.headers


# --- error paths are the ones a complaint is about ------------------------------------

def test_503_without_an_explainer_still_carries_the_id(make_client):
    r = _post(make_client(explainer=None))

    assert r.status_code == 503
    assert r.headers[REQUEST_ID_HEADER]


def test_422_on_malformed_metadata_still_carries_the_id(make_client):
    with make_client() as c:
        r = c.post(
            "/predict",
            files={"image": ("a.png", _png_bytes(), "image/png")},
            data={"metadata": "{not json"},
        )

    assert r.status_code == 422
    assert r.headers[REQUEST_ID_HEADER]


def test_error_response_echoes_a_supplied_id(make_client):
    """Correlation has to survive the failure, or it is useless where it is needed."""
    supplied = "trace-0f1e2d"
    r = _post(make_client(explainer=None), headers={REQUEST_ID_HEADER: supplied})

    assert r.status_code == 503
    assert r.headers[REQUEST_ID_HEADER] == supplied


def test_unhandled_500_still_carries_the_id():
    """The one response the middleware cannot reach on its own.

    `ServerErrorMiddleware` is hardcoded outside the user middleware stack and writes
    its 500 to its own `send`, so this is covered by the `Exception` handler instead.
    `raise_server_exceptions=False` is required: TestClient otherwise re-raises the
    exception (Starlette always re-raises after handling) and the response is never
    inspected.
    """
    app = create_app(ensemble=object(), explainer=None, client=None)

    @app.get("/_boom")
    async def boom() -> dict:
        raise RuntimeError("unhandled")

    with TestClient(app, raise_server_exceptions=False) as c:
        supplied = c.get("/_boom", headers={REQUEST_ID_HEADER: "boom-trace-1"})
        generated = c.get("/_boom")

    assert supplied.status_code == 500
    assert supplied.headers[REQUEST_ID_HEADER] == "boom-trace-1"
    # body stays Starlette's bare 500: an exception message could carry patient data
    assert supplied.json() == {"detail": "Internal Server Error"}
    assert generated.headers[REQUEST_ID_HEADER]


def test_unhandled_500_does_not_echo_a_hostile_id():
    """The error path must not become the one place validation is skipped."""
    app = create_app(ensemble=object(), explainer=None, client=None)

    @app.get("/_boom")
    async def boom() -> dict:
        raise RuntimeError("unhandled")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/_boom", headers={REQUEST_ID_HEADER: "abc\r\nInjected: evil"})

    assert r.status_code == 500
    assert "Injected" not in "\n".join(f"{k}: {v}" for k, v in r.headers.items())


def test_404_from_an_unrouted_path_still_carries_the_id(make_client):
    """A 404 is produced by Starlette's router, not by any handler of ours."""
    with make_client() as c:
        r = c.get("/no-such-endpoint")

    assert r.status_code == 404
    assert r.headers[REQUEST_ID_HEADER]


# --- handlers can reach the id --------------------------------------------------------

def test_id_is_available_on_request_state(make_client):
    """Logging correlation depends on this: /predict reads it off request.state."""
    app = create_app(ensemble=object(), explainer=None, client=None)

    seen = {}

    @app.get("/_probe")
    async def probe(request: Request) -> dict:
        seen["id"] = request.state.request_id
        return {"ok": True}

    with TestClient(app) as c:
        r = c.get("/_probe", headers={REQUEST_ID_HEADER: "state-check-1"})

    assert r.headers[REQUEST_ID_HEADER] == "state-check-1"
    assert seen["id"] == "state-check-1"


# --- the app must still boot with opentelemetry absent --------------------------------

def test_app_boots_and_correlates_without_opentelemetry(monkeypatch, make_client):
    """`observability` is an optional extra; request ids are core and must not need it."""
    import builtins

    real_import = builtins.__import__

    def no_otel(name, *args, **kwargs):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_otel)

    with make_client() as c:
        r = c.get("/health")

    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER]


# --- exactly one header, even if a handler sets its own -------------------------------

def test_header_is_replaced_not_duplicated():
    """A handler setting X-Request-ID itself must not produce two headers.

    The naive middleware appends, which emits the header twice when a handler already
    set one. A duplicated correlation header is worse than a wrong one: whoever reads
    the log cannot tell which value the client actually saw. No handler in this app does
    that today, so this guards a latent trap rather than a live bug -- and it is exactly
    the kind of thing a later `/explain` or `/history` handler could introduce.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from api.main import RequestIDMiddleware

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/sets-own")
    async def sets_own() -> JSONResponse:
        return JSONResponse({"ok": True}, headers={REQUEST_ID_HEADER: "handler-set-value"})

    with TestClient(app, raise_server_exceptions=False) as c:
        values = c.get("/sets-own").headers.get_list("x-request-id")

    assert len(values) == 1, f"expected exactly one request id header, got {values}"
    # the middleware's id wins, so the value in the log matches the value the client saw
    assert values[0] != "handler-set-value"
