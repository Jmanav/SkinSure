# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# SkinSure — AI-Powered Skin Disease Screening for Indian Skin Tones

> A clinical decision support system built specifically for Fitzpatrick IV–VI skin types, combining a deep learning ensemble, explainability tooling, and an agentic layer to guide patients from image capture to care.

---

## Current Implementation State

The rest of this file describes the full planned architecture. Before diving in, note what is **actually implemented** vs. what is planned:

**Fully implemented:**
- Pydantic schemas — `api/schemas/patient.py` (`PatientMetadata`) and `api/schemas/predictions.py` (`Predictions`, `XAISummary`, `ClinicalSummary`)
- Clinical summarizer — `src/agents/clinical_summarizer.py` (context builder + async Anthropic SDK call + post-processing)
- Config system — `src/utils/config.py` loads `configs/agents.yaml` via `get_settings()` (cached with `lru_cache`)
- `configs/agents.yaml` — sets the model (`claude-haiku-4-5-20251001`), `max_tokens`, and `temperature` for the summarizer

**Empty stubs (files exist, no logic):**
- `src/agents/orchestrator.py`, `severity_triage.py`, `treatment_advisor.py`, `doctor_referral.py`, `pharmacy_locator.py`
- `src/preprocessing/artifact_removal.py`
- `src/agents/tools/` — directory exists, no files

**Not yet created:**
- `api/main.py` and all API routes / middleware
- All other preprocessing modules (`quality_check.py`, `skin_tone_classifier.py`, `hair_removal.py`, `color_normalization.py`, `augmentation.py`)
- ML model Python files (only Jupyter notebooks exist in `src/models/`)
- `src/explainability/`, `src/federated/`, `training/`, `tests/`, `scripts/`, `docker/`, `.github/`
- `pyproject.toml` / `requirements.txt` — no package definition yet
- `.env.example`

---

## Commands

No package manager or test runner is configured yet. Once `pyproject.toml` is added, commands will be:

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/unit/test_clinical_summarizer.py

# Run with fairness audit
pytest tests/fairness/test_skin_tone_bias.py

# Start API dev server (once api/main.py exists)
uvicorn api.main:app --reload

# Training
python training/train.py --config training/configs/ensemble.yaml
python training/evaluate.py --fairness-audit
```

---

## Architecture and Key Patterns

### Schema layer (`api/schemas/`)

`PatientMetadata` enforces the Fitzpatrick IV–VI gate at the boundary — any image from a Fitzpatrick I–III patient is rejected with a clear validator error before reaching the model. `ClinicalSummary` has a `@model_validator` that enforces `urgent → immediate_referral` at the Pydantic level, so the safety constraint is checked in two places: the LLM system prompt and the schema validator.

`Predictions.confidence_tier()` maps `agreement_score` to `"high" / "moderate" / "low"` (≥0.80 / ≥0.70 / <0.70) — use this method rather than replicating the threshold logic.

### Clinical summarizer (`src/agents/clinical_summarizer.py`)

Two-phase design — `build_context()` is pure Python (no LLM), `summarize()` is async and calls the Anthropic API. Post-processing in `summarize()` runs after the LLM response:
1. If `xai_summary.artifact_flag`: downgrade `confidence_level` one tier via `_TIER_DOWN` and force `recapture_needed = True`.
2. If `agreement_score < 0.70` and no caveat already mentions "model disagreement": append the standard caveat string.

The LLM is instructed to return **JSON only** — no markdown, no preamble. Parsing failures raise `ValueError` with the raw response included for debugging.

### Config system (`src/utils/config.py`)

`get_settings()` is `lru_cache`-wrapped — call it freely; it reads `configs/agents.yaml` only once. `Settings` is a `BaseSettings` subclass, so environment variables can override YAML values if needed.

### Design invariants

- **LLM interprets; Python decides.** The LLM proposes severity; deterministic rules in `severity_triage.py` (once implemented) override it. Never let the LLM have the final word on urgent/non-urgent.
- **Context builder is deterministic; summarizer is not.** All clinical reasoning lives inside the LLM with a constrained system prompt. `build_context()` only formats — no interpretation.
- **Fairness is a first-class CI check.** Per-Fitzpatrick-subtype accuracy must be tested on every model update.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Agent | Anthropic Claude (via `anthropic` async SDK) |
| API | FastAPI + Pydantic v2 |
| Config | YAML + `pydantic-settings` |
| ML Frameworks | PyTorch, timm (planned) |
| Model Backbones | EfficientNet, ConvNeXt, Swin Transformer (planned) |
| xAI | Grad-CAM, SHAP, LIME (planned) |
| Federated Learning | Custom FedAvg + OpenDP (planned) |
| Containerisation | Docker + docker-compose (planned) |
| CI/CD | GitHub Actions (planned) |

---

## Full Architecture Reference

### System Architecture

```
User (Mobile App)
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
| Quality check | `quality_check.py` | Reject blurry or low-resolution images |
| Skin tone gate | `skin_tone_classifier.py` | Verify Fitzpatrick IV–VI; flag outliers |
| Hair removal | `hair_removal.py` | DullRazor algorithm + inpainting |
| Artifact removal | `artifact_removal.py` | Remove ruler marks, ink, air bubbles |
| Color normalization | `color_normalization.py` | Shades-of-Gray / Macenko / Reinhard |
| Augmentation | `augmentation.py` | Tone-aware augmentation for training |

Ensemble voting: weighted average across EfficientNet (texture), ConvNeXt (varied lighting), Swin Transformer (global lesion context). `agreement_score` between the three is computed and passed to the summarizer — low agreement triggers explicit uncertainty flagging.

### Agentic Layer

| Agent | File | Status |
|---|---|---|
| Orchestrator | `src/agents/orchestrator.py` | Stub |
| Clinical Summarizer | `src/agents/clinical_summarizer.py` | **Implemented** |
| Severity Triage | `src/agents/severity_triage.py` | Stub |
| Treatment Advisor | `src/agents/treatment_advisor.py` | Stub |
| Doctor Referral | `src/agents/doctor_referral.py` | Stub |
| Pharmacy Locator | `src/agents/pharmacy_locator.py` | Stub |

Severity triage hard overrides (to implement): melanoma at any moderate+ confidence → `urgent`; model confidence < 40% → block, request re-capture; rapidly spreading infection → `urgent`.

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

### Federated Learning (`src/federated/` — planned)

Hospital partners run local training; only model weight deltas (protected by differential privacy) are transmitted to the central server. Required for DPDP Act compliance. Components: `client.py`, `server.py` (FedAvg), `privacy.py` (OpenDP), `communication.py`.

### Testing (`tests/` — planned)

```bash
pytest tests/unit/          # Preprocessing, model, agent, explainability
pytest tests/integration/   # End-to-end: image in → ClinicalSummary out
pytest tests/fairness/test_skin_tone_bias.py  # Per-Fitzpatrick accuracy audit
```

Fairness suite runs automatically on every PR — any regression in per-tone accuracy fails the check.

---

*For architecture deep-dives, see `docs/architecture.md`. For domain knowledge on Indian skin tones and regional conditions, see `docs/indian_skin_tones.md`.*
