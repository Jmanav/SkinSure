from __future__ import annotations

import asyncio
import io
import logging
import re
import uuid
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import ScreeningStatus
from api.schemas.responses import PredictResponse
from src.agents.orchestrator import Explainer, Predictor, run_screening
from src.explainability.explainer import GradCamExplainer

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_ID_LEN = 128
# Full-match allowlist, not a sanitiser. A sanitiser has to decide what to keep, and
# every "strip the bad bytes" rule is one missed codepoint away from putting CR, LF or
# NUL into a response header or a log line (header splitting / log forging). Replacing
# the whole value instead makes that structurally impossible: anything outside this set
# never reaches a header or a log at all. The set is deliberately narrower than RFC 7230
# token chars -- uuid4().hex, ULIDs and the id formats gateways emit all fit, and a
# caller using something exotic loses correlation, not the request.
_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,%d}\Z" % MAX_REQUEST_ID_LEN)


def _resolve_request_id(incoming: str | None) -> str:
    """Reuse a caller's request id when it is safe to echo, otherwise mint a fresh one.

    Reuse is what lets a gateway or client correlate its own log line with ours. A
    rejected value is replaced rather than 400'd: the id is diagnostic metadata, and
    failing a clinical screening request over a malformed trace header would trade a
    real outcome for a cosmetic one.
    """
    if incoming is not None and _REQUEST_ID_RE.match(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestIDMiddleware:
    """Attach a request id to every request and echo it on every response.

    Written as raw ASGI rather than with `@app.middleware("http")` because the header is
    set at the `send` boundary: every response the app produces -- 200, the 422 from the
    metadata gate, the 400/413 upload guards, the 503 with no explainer, the router's own
    404 -- goes through one line of code, instead of each handler remembering to add it.

    It does NOT cover an *unhandled* exception. Starlette hardcodes
    `ServerErrorMiddleware` outside the user middleware stack and that 500 is written to
    its `send`, not ours, no matter how this middleware is installed. The
    `Exception` handler in `create_app` covers that case using the id this middleware has
    already put on the scope; verified by test, since the alternative is that the only
    uncorrelated response is the worst failure we have.

    The id is stashed on `request.state` (via the raw ASGI scope, which is the dict
    `Request.state` reads) so handlers and log records reach it without plumbing it
    through signatures.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for name, value in scope.get("headers", ()):
            if name.lower() == b"x-request-id":
                # latin-1 is the HTTP header wire encoding; never raises, so a
                # non-UTF-8 header cannot 500 the request before validation sees it.
                incoming = value.decode("latin-1")
                break

        request_id = _resolve_request_id(incoming)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                # REPLACE, don't append. If a handler sets X-Request-ID itself the naive
                # append emits the header twice (verified: two values on one response),
                # and a duplicated correlation header is worse than a wrong one -- whoever
                # reads the log cannot tell which value the client actually saw. No handler
                # does this today, so this is a latent trap rather than a live bug; drop
                # any existing copy first so there is exactly one, always ours.
                headers = [
                    (name, value)
                    for name, value in message.get("headers", ())
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_id)


def _load_ensemble() -> Predictor:
    # Deferred import: pulls in torch and loads three checkpoints.
    from src.models.ensemble import SkinEnsemble
    from src.utils.config import get_checkpoint_paths, get_model_config

    w = get_model_config().weights
    return SkinEnsemble(
        get_checkpoint_paths(), weights=(w["efficientnet"], w["convnext"], w["swin"])
    )


def create_app(
    ensemble: Predictor | None = None,
    explainer: Explainer | None = None,
    client: anthropic.AsyncAnthropic | None = None,
) -> FastAPI:
    """Build the app. Dependencies are injectable so tests never load checkpoints or call the API.

    `explainer` defaults to `GradCamExplainer` when the ensemble is a real `SkinEnsemble`.
    If no explainer can be built, /predict answers 503 rather than inventing explainability
    data for a clinical summary.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = ensemble or await asyncio.to_thread(_load_ensemble)
        app.state.ensemble = loaded
        # Default: Grad-CAM on the real ensemble. Test doubles without `backbones` get none,
        # so /predict still answers 503 for them unless an explainer is injected.
        if explainer is None and hasattr(loaded, "backbones"):
            app.state.explainer = GradCamExplainer(loaded)
        else:
            app.state.explainer = explainer
        app.state.client = client
        yield

    app = FastAPI(title="SkinSure", lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Carry the id onto the one response the middleware provably cannot reach.

        Starlette hardcodes `ServerErrorMiddleware` *outside* the user middleware stack
        (`build_middleware_stack`), and it writes the 500 to its own `send`, not ours --
        so without this handler an unhandled exception is the single response with no
        `X-Request-ID`, which is precisely the response someone files a complaint about.
        Registering a handler for `Exception` makes it this function's response instead,
        and the id is still on the scope because the middleware wrote it before the
        request ever reached a handler. Starlette re-raises afterwards either way, so the
        server still logs the traceback; only the client-visible body is ours. Status and
        body deliberately match Starlette's bare 500 -- no detail, since an exception
        message here could carry patient data.
        """
        request_id = getattr(request.state, "request_id", None)
        # The id plus the exception type only -- no traceback and no `str(exc)`. Starlette
        # re-raises, so the server logs the full traceback anyway; repeating it here would
        # double every 500 in the log, and an exception message can carry patient data.
        logger.error(
            "unhandled error (request_id=%s, error_type=%s)",
            request_id or "unknown",
            type(exc).__qualname__,
        )
        headers = {REQUEST_ID_HEADER: request_id} if request_id else None
        return JSONResponse(
            {"detail": "Internal Server Error"}, status_code=500, headers=headers
        )

    @app.get("/health")
    async def health(request: Request) -> dict:
        return {
            "status": "ok",
            "explainer_ready": request.app.state.explainer is not None,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        request: Request,
        image: UploadFile = File(...),
        metadata: str = Form(..., description="PatientMetadata as a JSON string"),
    ) -> PredictResponse:
        state = request.app.state
        if state.explainer is None:
            raise HTTPException(503, "Explainability module is not configured on this server.")

        try:
            patient = PatientMetadata.model_validate_json(metadata)
        except ValidationError as exc:
            # includes the Fitzpatrick IV-VI gate; errors describe fields, not the raw input
            raise HTTPException(
                422,
                detail=exc.errors(include_input=False, include_url=False, include_context=False),
            )

        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            pil = Image.open(io.BytesIO(data))
            pil.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "Could not read the uploaded file as an image.")

        try:
            result = await run_screening(
                pil,
                patient,
                ensemble=state.ensemble,
                explainer=state.explainer,
                client=state.client,
            )
        except (ValueError, anthropic.APIError):
            # parse/schema failure or upstream LLM error; details (incl. raw output) stay in logs.
            # The request id goes in the message so a user complaint ("it failed, here is
            # the id my client showed") maps to this line -- the id is the only handle,
            # since nothing patient-identifying may be logged.
            logger.exception(
                "summarizer failed (request_id=%s)",
                getattr(request.state, "request_id", "unknown"),
            )
            raise HTTPException(502, "The summary service failed; please try again.")

        completed = result.status is ScreeningStatus.COMPLETED
        return PredictResponse(
            status=result.status,
            summary=result.summary,
            predictions=result.predictions if completed else None,
            messages=result.messages,
            warnings=result.warnings,
            escalated=result.escalated,
            treatment_advice_allowed=result.treatment_advice_allowed,
        )

    return app


app = create_app()
