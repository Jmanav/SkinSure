import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeClient, llm_json
from PIL import Image

from api.main import MAX_UPLOAD_BYTES, create_app
from api.schemas.predictions import DiseaseClass

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
    def _make(predictions=None, llm_text=None, explainer="default"):
        if explainer == "default":
            xai = make_xai()
            explainer = lambda tensor, preds: xai  # noqa: E731
        app = create_app(
            ensemble=FakeEnsemble(predictions or make_predictions()),
            explainer=explainer,
            client=FakeClient(llm_text or llm_json()),
        )
        return TestClient(app)

    return _make


def _post(client, image=None, patient=None, **kwargs):
    files = {"image": ("lesion.png", image if image is not None else _png_bytes(), "image/png")}
    data = {"metadata": json.dumps(patient or PATIENT)}
    with client as c:
        return c.post("/predict", files=files, data=data, **kwargs)


def test_health(make_client):
    with make_client() as c:
        assert c.get("/health").json() == {"status": "ok", "explainer_ready": True}


def test_health_reports_missing_explainer(make_client):
    with make_client(explainer=None) as c:
        assert c.get("/health").json()["explainer_ready"] is False


def test_predict_returns_clinical_summary(make_client):
    r = _post(make_client())

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["summary"]["severity"] == "routine"
    assert body["predictions"]["primary_label"] == "Eczema"
    assert body["treatment_advice_allowed"] is True


def test_response_does_not_leak_patient_name(make_client):
    assert "Asha" not in _post(make_client()).text


def test_melanoma_is_escalated_to_urgent(make_client, make_predictions):
    preds = make_predictions(
        primary_label=DiseaseClass.MELANOMA,
        primary_confidence=0.7,
        secondary_label=DiseaseClass.KERATOSIS,
    )
    body = _post(make_client(preds, llm_json(severity="routine"))).json()

    assert body["summary"]["severity"] == "urgent"
    assert body["summary"]["immediate_referral"] is True
    assert body["escalated"] is True
    assert body["treatment_advice_allowed"] is False


def test_low_confidence_is_blocked_without_summary(make_client, make_predictions):
    preds = make_predictions(primary_confidence=0.3, secondary_confidence=0.2)
    body = _post(make_client(preds)).json()

    assert body["status"] == "blocked"
    assert body["summary"] is None and body["predictions"] is None
    assert body["messages"]


def test_small_image_is_rejected_with_reasons(make_client):
    body = _post(make_client(), image=_png_bytes(size=64)).json()

    assert body["status"] == "rejected"
    assert "too small" in body["messages"][0]


def test_fitzpatrick_gate_returns_422(make_client):
    r = _post(make_client(), patient={**PATIENT, "fitzpatrick_type": "II"})

    assert r.status_code == 422
    assert "Asha" not in r.text


def test_invalid_metadata_json_returns_422(make_client):
    with make_client() as c:
        r = c.post(
            "/predict",
            files={"image": ("a.png", _png_bytes(), "image/png")},
            data={"metadata": "{not json"},
        )
    assert r.status_code == 422


def test_non_image_upload_returns_400(make_client):
    r = _post(make_client(), image=b"definitely not an image")
    assert r.status_code == 400


def test_oversized_upload_returns_413(make_client):
    r = _post(make_client(), image=b"0" * (MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413


def test_missing_explainer_returns_503(make_client):
    r = _post(make_client(explainer=None))
    assert r.status_code == 503


def test_unparseable_llm_output_returns_502_without_raw_text(make_client):
    r = _post(make_client(llm_text="the raw model output SECRET"))

    assert r.status_code == 502
    assert "SECRET" not in r.text
