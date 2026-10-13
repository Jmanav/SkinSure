"""End-to-end with the REAL checkpoints and real Grad-CAM; only the LLM is faked.

Skipped when the (gitignored) checkpoint files are not present, e.g. in CI.
"""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeClient, llm_json
from PIL import Image

from api.main import create_app
from src.utils.config import get_checkpoint_paths

paths = get_checkpoint_paths()
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (paths.efficientnet, paths.convnext, paths.swin)),
    reason="model checkpoints not available",
)

PATIENT = dict(
    patient_id=1, patient_name="Asha Verma", patient_age=45, patient_sex="female",
    fitzpatrick_type="V", body_area="arm", symptom_duration_days=30,
)


def _png() -> bytes:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_default_app_uses_real_ensemble_and_gradcam():
    app = create_app(client=FakeClient(llm_json()))   # no ensemble / explainer injected
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "explainer_ready": True}

        r = c.post(
            "/predict",
            files={"image": ("lesion.png", _png(), "image/png")},
            data={"metadata": json.dumps(PATIENT)},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"completed", "blocked"}
    if body["status"] == "completed":
        assert body["predictions"]["primary_label"] in {
            "Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea",
        }
    assert "Asha" not in r.text
