# CLAUDE.md

---

# SkinSure — AI-Powered Skin Disease Screening for Indian Skin Tones

> A clinical decision support system built specifically for Fitzpatrick IV–VI skin types, combining a deep learning ensemble, explainability tooling, and an agentic layer to guide patients from image capture to care.

---

## Documentation map (keep these consistent)

| File | Role |
|---|---|
| `CLAUDE.md` (this file) | **Source of truth** for current state, conventions and design invariants |
| `docs/architecture.md` | Target folder layout and import map |
| `docs/ROADMAP.md` | Phase-by-phase build plan (phases 0–12) |
| `fedler.md` | **Source of truth for federated learning and differential privacy** (overrides any other doc) |
| `src/models/CLAUDE.md` | Model notebooks, training decisions, checkpoint format |
| `QnA.md` | Learning log; not normative |

If a doc disagrees with this file or with `fedler.md`, fix the doc. When code changes the state below, update this file in the same change.

---

## Current Implementation State

Last audited 2026-10-10 against the files on disk.

**Implemented:**
- Schemas — `api/schemas/patient.py` (`PatientMetadata`, Fitzpatrick IV–VI gate) and `api/schemas/predictions.py` (`Predictions`, `XAISummary`, `Severity`, `ConfidenceLevel`, `ClinicalSummary`; validator enforces `urgent → immediate_referral`)
- Model layer — `src/models/ensemble.py` (`SkinEnsemble`, `CheckpointPaths`), `efficientnet.py`, `convnext.py`, `swin.py`; checkpoints in `models/checkpoints/{efficientNet,convNext,swin}/`; training/eval notebooks in `src/models/`
- Preprocessing — `src/preprocessing/`: `inference_transform.py` (224px resize + ImageNet normalisation, `preprocess_image`), `quality_check.py`, `hair_removal.py`, `artifact_removal.py`, `color_normalization.py`, `pipeline.py` (`preprocess()`), thresholds and step switches in `configs/preprocessing.yaml`
- `scripts/predict_one.py` — single-image inference through the preprocessing pipeline (works)
- Clinical summarizer — `src/agents/clinical_summarizer.py` (`build_context`, async `summarize`, post-processing); demo in `scripts/summarize_one.py`
- Severity triage — `src/agents/severity_triage.py` (`triage(predictions, summary, patient) -> TriageResult`; all three hard overrides)
- Orchestrator and API — `src/agents/orchestrator.py` (`run_screening(image, patient, *, ensemble, explainer, client) -> ScreeningResult`: quality gate → ensemble → explainer → summarizer → triage; statuses `completed` / `rejected` / `blocked`; `treatment_advice_allowed` is False for urgent, referral and re-capture cases) and `api/main.py` (`create_app(ensemble, explainer, client)`; `GET /health`, `POST /predict` taking a multipart `image` + `metadata` JSON string). The **explainer is injected and has no default**: Grad-CAM lives outside the repo for now, so `/predict` returns 503 until one is passed to `create_app`.
- Config — `src/utils/config.py` (`get_settings()` for `configs/agents.yaml`; `get_model_config()` / `get_checkpoint_paths()` for `configs/models.yaml`; `get_preprocessing_config()` for `configs/preprocessing.yaml`), `configs/agents.yaml`, `configs/models.yaml` (checkpoint paths and ensemble weights)
- Packaging — `pyproject.toml`, `.env.example`, `.gitignore`

**Known gaps:**
1. `tests/unit/` covers schemas, severity triage (every override branch), the clinical summarizer (fake client), the orchestrator and preprocessing; `tests/integration/` covers `/predict` and `/health` with fakes. 131 tests pass. No `tests/fairness/` yet. Nothing in the suite touches the real LLM, a real photo, or real-data accuracy.
2. `summarize_one.py` has not been run against the real Anthropic API.
3. Hair and artifact removal are untuned: on 100 unmodified Basal_Cell_Carcinoma images they fired on 33% (hair) and 47% (artifacts); on a test image with dark corners the hair mask covered 57% and the step skipped itself. Both stay off until tuned and compared on a held-out set (only BCC was measured; the run did not finish for other classes).
4. Preprocessing is unit-tested on synthetic images only (gate boundaries, guards, step order, defaults off). Those tests pin behaviour; they say nothing about tuning on real images (see gap 3).
5. `predict_one.py` was run once on a synthetic image (`scratch_test_image.png`): checkpoints load and it prints predictions. Not yet checked on a real lesion photo.

**Empty stubs (0 bytes):** `treatment_advisor.py`, `doctor_referral.py`, `pharmacy_locator.py`; `src/CLAUDE.md`, `src/agents/CLAUDE.md`; `docs/indian_skin_tones.md` (domain knowledge still to be written). `src/agents/tools/` is an empty directory.

**Not yet created:**
- `GET /history`, `POST /history`, `POST /explain`; auth, rate limiting and middleware
- Preprocessing: `augmentation.py` (training only)
- `src/explainability/`, `training/`, `docker/`, `.github/`
- other planned scripts

**Future scope (deliberately not planned for the core product):**
- `skin_tone_classifier.py` — automatic Fitzpatrick detection from the image. Today the IV–VI gate relies only on the Fitzpatrick type in `PatientMetadata`.
- Federated learning and differential privacy — deferred entirely; the plan lives in `fedler.md`.

---

## Commands

`pyproject.toml` exists (install with `pip install -e .[dev]`), and `tests/unit/` + `tests/integration/` run today. Commands (some not available yet):

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/unit/test_clinical_summarizer.py

# Run with fairness audit
pytest tests/fairness/test_skin_tone_bias.py

# Start API dev server (/predict answers 503 until an explainer is supplied)
uvicorn api.main:app --reload

# Training (top-level training/ directory, once created)
python training/train.py --config training/configs/ensemble.yaml
python training/evaluate.py --fairness-audit
```

Works today:

```bash
python scripts/predict_one.py <image>     # ensemble prediction
python scripts/summarize_one.py          # needs ANTHROPIC_API_KEY in .env; calls the real API
```

---

## Architecture and Key Patterns

### Schema layer (`api/schemas/`)

`PatientMetadata` enforces the Fitzpatrick IV–VI gate at the boundary — any image from a Fitzpatrick I–III patient is rejected with a clear validator error before reaching the model. Fields: `patient_id`, `patient_name`, `patient_age` (0–120), `patient_sex`, `fitzpatrick_type`, `patient_location` (optional), `current_season` (optional), `body_area`, `symptom_duration_days`, `medical_history` (list), `is_spreading_rapidly` (bool, default False). Enums: `Sex`, `Season`, `BodyArea`, `FitzpatrickType`.

`ClinicalSummary` fields: `severity` (`routine` / `monitor` / `urgent`), `summary`, `recommended_action`, `confidence_level` (`high` / `moderate` / `low`), `confidence_caveats`, `recapture_needed`, `immediate_referral`. Its `@model_validator` enforces `urgent → immediate_referral` at the Pydantic level, so the safety constraint is checked in two places: the LLM system prompt and the schema validator.

`Predictions` labels are the `DiseaseClass` enum (exact checkpoint strings: `Basal_Cell_Carcinoma`, `Eczema`, `Keratosis`, `Melanoma`, `Tinea`); confidences and `agreement_score` are bounded 0–1, and the secondary label must differ from the primary with confidence ≤ primary. `SkinEnsemble` refuses checkpoints whose classes are not exactly that set.

`XAISummary` fields: `focus_description` (`FocusRegion` enum: `lesion_centre`, `lesion_border`, `diffuse`, `background_skin`, `artifact`), `boundary_alignment` (`BoundaryAlignment`: `good`, `partial`, `poor`), `artifact_flag`. A validator requires `artifact_flag=True` when focus is `artifact`.

`Predictions.confidence_tier()` maps `agreement_score` (a fraction of 3 models: 0 / 0.33 / 0.67 / 1.0) to `"high"` (all three agree) / `"moderate"` (two agree) / `"low"` (otherwise) — use this method rather than replicating the threshold logic.

### Clinical summarizer (`src/agents/clinical_summarizer.py`)

Two-phase design — `build_context()` is pure Python (no LLM, and it deliberately omits `patient_name`), `summarize()` is async and calls the Anthropic API (model, `max_tokens`, `temperature` come from `configs/agents.yaml`; the client is injectable for tests). Post-processing in `summarize()` runs after the LLM response:
1. If `xai_summary.artifact_flag`: downgrade `confidence_level` one tier via `_TIER_DOWN` and force `recapture_needed = True`.
2. If `confidence_tier() == "low"` and no caveat already mentions "model disagreement": append the standard caveat string.

The LLM is instructed to return **JSON only** — no markdown, no preamble. Parsing or schema failures raise `ValueError` with the raw response included for debugging.

### Config system (`src/utils/config.py`)

`get_settings()` and `get_model_config()` are `lru_cache`-wrapped — call them freely; each reads its YAML only once. `get_checkpoint_paths()` returns a `CheckpointPaths` resolved against the project root (it imports `ensemble` lazily because that pulls in torch). `Settings` is a `BaseSettings` subclass, so environment variables can override YAML values if needed.

### Design invariants

- **LLM interprets; Python decides.** The LLM proposes severity; deterministic rules in `severity_triage.py` override it. Never let the LLM have the final word on urgent/non-urgent.
- **Context builder is deterministic; summarizer is not.** All clinical reasoning lives inside the LLM with a constrained system prompt. `build_context()` only formats — no interpretation.
- **Fairness is a first-class CI check.** Per-Fitzpatrick-subtype accuracy must be tested on every model update.
- **Uploaded images are never used for training.** They are kept only for the patient's own history (`GET /history`).
- **Secrets never touch git or images.** `.env` locally (gitignored), a secret manager in the cloud.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Agent | Anthropic Claude (via `anthropic` async SDK) |
| API | FastAPI + Pydantic v2 |
| Config | YAML + `pydantic-settings` |
| ML Frameworks | PyTorch, timm |
| Model Backbones | EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny |
| xAI | Grad-CAM first; SHAP, LIME later (planned) |
| Federated learning / DP | Deferred — see `fedler.md` |
| Containerisation | Docker + docker-compose (planned) |
| CI/CD | GitHub Actions (planned) |

---

## Full Architecture Reference

### System Architecture

```
User (Web App)
      │
      ▼
[FastAPI Backend]
      │
      ├── POST /predict ──► [Preprocessing Pipeline]
      │                           │
      │                    [Ensemble Model]
      │                    EfficientNet + ConvNeXt + Swin
      │                           │
      │                    [xAI Layer]
      │                    Grad-CAM / SHAP / LIME
      │                           │
      │                    [Orchestrator Agent]
      │                           │
      │              ┌────────────┼────────────┐
      │              ▼            ▼            ▼
      │     [Clinical        [Severity     [Treatment
      │      Summarizer]      Triage]       Advisor]
      │              │            │            │
      │              └────────────┼────────────┘
      │                           ▼
      │                   [Router]
      │              ┌────────────┴────────────┐
      │              ▼                         ▼
      │     [Pharmacy Locator]        [Doctor Referral]
      │
      └── GET /history ──► [Patient Records]
```

### ML Pipeline

| Step | Module | Purpose |
|---|---|---|
| Quality check | `quality_check.py` | Reject images under 200px (short side) or nearly featureless; warn on borderline blur |
| Hair removal | `hair_removal.py` | DullRazor-style black-hat + inpainting (off by default) |
| Artifact removal | `artifact_removal.py` | Inpaint pen ink, bubbles/glare, ruler lines (off by default) |
| Color normalization | `color_normalization.py` | Shades-of-Gray (off by default) |
| Augmentation | `augmentation.py` | Tone-aware augmentation for training |

`preprocess(image)` (in `pipeline.py`) orients the image (EXIF), runs the quality gate, then the enabled pixel-changing steps in the order hair → artifacts → colour, and returns a `PreprocessResult` (`image`, `quality`, `steps_applied`, `notes`). A rejected image comes back with `quality.ok == False` and patient-readable `quality.reasons`; borderline blur adds a `quality.warnings` entry instead. **The models were trained with only resize + ImageNet normalisation, so all three pixel-changing steps are off by default** (`configs/preprocessing.yaml`); enable one only after comparing predictions with and without it on a held-out set. Thresholds come from a 1,500-image sample of the raw dataset: short side ≥ 200px; blur (Laplacian variance) rejects below 5 and warns below 15. A higher blur cut-off would reject BCC and Melanoma images far more often than Eczema or Tinea (about 18% / 9% vs 1% / 0.3% at 15), so it is kept low. Shades-of-Gray can flatten genuine skin-tone colour, which matters for Fitzpatrick IV–VI.

The Fitzpatrick IV–VI gate is enforced by `PatientMetadata`; an image-based `skin_tone_classifier.py` is future scope.

Ensemble voting: weighted average across EfficientNet (texture), ConvNeXt (varied lighting), Swin Transformer (global lesion context), with weights from `configs/models.yaml`. `agreement_score` = fraction of the three models whose top-1 matches the ensemble top-1; it is passed to the summarizer, and low agreement triggers explicit uncertainty flagging.

Classes (5), in model index order (alphabetical, as stored in each checkpoint's `classes`): Basal_Cell_Carcinoma, Eczema, Keratosis, Melanoma, Tinea. See `src/models/CLAUDE.md` for training details and sensitivity targets.

### Agentic Layer

| Agent | File | Status |
|---|---|---|
| Orchestrator | `src/agents/orchestrator.py` | **Implemented** (`run_screening`; router to later agents still to come) |
| Clinical Summarizer | `src/agents/clinical_summarizer.py` | **Implemented** |
| Severity Triage | `src/agents/severity_triage.py` | **Implemented** |
| Treatment Advisor | `src/agents/treatment_advisor.py` | Stub |
| Doctor Referral | `src/agents/doctor_referral.py` | Stub |
| Pharmacy Locator | `src/agents/pharmacy_locator.py` | Stub |

Severity triage hard overrides (`triage(predictions, summary, patient) -> TriageResult`; thresholds 0.40 in `severity_triage.py`): melanoma at any moderate+ confidence → `urgent`; model confidence < 40% → block, request re-capture; rapidly spreading infection (`patient.is_spreading_rapidly` and an infection class — currently Tinea) → `urgent`. Rules only escalate or block, never downgrade; escalation and the infection rule are skipped when the result is blocked. Urgent cases must never receive self-treatment advice (the router suppresses the treatment advisor).

### xAI (`src/explainability/` — planned)

- **Grad-CAM / Grad-CAM++** — spatial heatmap converted to `XAISummary` (focus, boundary alignment, artifact flag) before reaching the summarizer
- **SHAP** — feature-level patch attribution
- **LIME** — superpixel local explanations; validates model attends to lesion, not background

### API endpoints (planned)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Image + metadata → `ClinicalSummary` |
| `POST` | `/explain` | Prediction → xAI heatmap report |
| `GET` | `/history` | Retrieve patient history |
| `POST` | `/history` | Save consultation record |
| `GET` | `/health` | Service health check |

### Federated learning — deferred

Not part of the current build. The plan (hospital/clinic clients, Flower FedAvg, simulation first) lives in `fedler.md`, the source of truth for it.

### Testing (`tests/` — planned)

```bash
pytest tests/unit/          # Preprocessing, model, agent, explainability
pytest tests/integration/   # End-to-end: image in → ClinicalSummary out
pytest tests/fairness/test_skin_tone_bias.py  # Per-Fitzpatrick accuracy audit
```

Fairness suite runs automatically on every PR — any regression in per-tone accuracy fails the check.

---

*For the target layout see `docs/architecture.md`; for the build plan see `docs/ROADMAP.md`; for federated learning see `fedler.md`. Domain knowledge on Indian skin tones and regional conditions is still to be written in `docs/indian_skin_tones.md`.*
