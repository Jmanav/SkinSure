from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from api.schemas.patient import PatientMetadata
from api.schemas.predictions import ScreeningStatus
from api.schemas.responses import PredictResponse
from src.agents.orchestrator import Explainer, Predictor, run_screening

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


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

    `explainer` has no default: until the Grad-CAM implementation is supplied, /predict
    answers 503 rather than inventing explainability data for a clinical summary.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ensemble = ensemble or await asyncio.to_thread(_load_ensemble)
        app.state.explainer = explainer
        app.state.client = client
        yield

    app = FastAPI(title="SkinSure", lifespan=lifespan)

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
            # parse/schema failure or upstream LLM error; details (incl. raw output) stay in logs
            logger.exception("summarizer failed")
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
