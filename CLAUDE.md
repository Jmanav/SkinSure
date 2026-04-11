# SkinSure — AI-Powered Skin Disease Screening for Indian Skin Tones

> A clinical decision support system built specifically for Fitzpatrick IV–VI skin types, combining a deep learning ensemble, explainability tooling, and an agentic layer to guide patients from image capture to care.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [ML Pipeline](#4-ml-pipeline)
5. [Agentic Layer](#5-agentic-layer)
6. [Explainability](#6-explainability)
7. [API](#7-api)
8. [Federated Learning](#8-federated-learning)
9. [Project Structure](#9-project-structure)
10. [Tech Stack](#10-tech-stack)
11. [Getting Started](#11-getting-started)
12. [Testing & Fairness](#12-testing--fairness)
13. [Deployment](#13-deployment)

---

## 1. Project Overview

SkinSure is a mobile-first AI screening tool that helps patients in India identify skin conditions from a photograph. The system routes patients to the right level of care — self-care, routine dermatologist, or emergency referral — without ever stating a definitive diagnosis.

**Why India, why now?**

- India has a severe shortage of dermatologists (~1 per 100,000 population in rural areas)
- Existing dermatology AI datasets are heavily biased toward Fitzpatrick I–III (lighter skin tones), making most commercial tools unreliable for Indian patients
- Regional and seasonal epidemiology (monsoon fungal outbreaks, regional prevalence of conditions like tinea versicolor, vitiligo) significantly affects clinical likelihood

SkinSure addresses these gaps through tone-aware preprocessing, a fairness-audited ensemble, and India-specific clinical context baked into its agent prompts.

---

## 2. Key Features

- **Ensemble ML model** — EfficientNet + ConvNeXt + Swin Transformer with weighted voting
- **Tone-aware preprocessing** — Fitzpatrick IV–VI gate, color normalization, hair removal, artifact detection
- **xAI heatmaps** — Grad-CAM, SHAP, and LIME for spatial and feature-level explainability
- **Clinical summarizer agent** — LLM-powered agent that translates model output into plain-language patient summaries
- **Severity triage** — Rule-based hard overrides ensure urgent cases (e.g., suspected melanoma) are never downgraded
- **Downstream routing** — Pharmacy locator for self-care, specialist matching for referrals
- **Federated learning** — Privacy-preserving model updates from hospital partners without raw data leaving their premises
- **FastAPI backend** — REST endpoints for prediction, explanation, and patient history
- **Fairness auditing** — Per-skin-tone accuracy breakdowns in CI pipeline

---

## 3. System Architecture

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

**Data flow summary:**
1. Patient uploads a skin image with metadata (age, region, symptom duration, etc.)
2. Preprocessing cleans, normalizes, and validates the image
3. The ensemble produces top-k predictions with confidence scores and model agreement
4. xAI generates a heatmap and structured spatial description
5. The orchestrator agent coordinates the clinical summarizer and severity triage
6. The router dispatches to pharmacy locator or doctor referral based on severity
7. A structured `ClinicalSummary` is returned to the patient

---

## 4. ML Pipeline

### 4.1 Preprocessing (`src/preprocessing/`)

Every image passes through a deterministic pipeline before inference:

| Step | Module | Purpose |
|---|---|---|
| Quality check | `quality_check.py` | Reject blurry or low-resolution images |
| Skin tone gate | `skin_tone_classifier.py` | Verify Fitzpatrick IV–VI; flag outliers |
| Hair removal | `hair_removal.py` | DullRazor algorithm + inpainting |
| Artifact removal | `artifact_removal.py` | Remove ruler marks, ink, air bubbles |
| Color normalization | `color_normalization.py` | Shades-of-Gray / Macenko / Reinhard |
| Augmentation | `augmentation.py` | Tone-aware augmentation for training |

### 4.2 Ensemble Model (`src/models/`)

Three complementary backbones vote on each prediction:

- **EfficientNet** — Efficient convolutional feature extraction; strong on texture
- **ConvNeXt** — Modern CNN with transformer-inspired design; robust on varied lighting
- **Swin Transformer** — Hierarchical vision transformer; strong on global lesion context

The `ensemble.py` module implements weighted averaging with a tunable meta-learner. Agreement score between the three models is computed and passed downstream — low agreement triggers explicit uncertainty flagging in the clinical summary.

**Loss functions (`losses.py`):**
- Focal loss to handle class imbalance across rare conditions
- Class-balanced loss calibrated to Indian prevalence data

**Metrics (`metrics.py`):**
- Per-class accuracy, AUC-ROC
- Fairness metrics: accuracy breakdown by Fitzpatrick subtype

### 4.3 Training (`training/`)

Each backbone has its own YAML config (`configs/efficientnet.yaml`, etc.) controlling learning rate, augmentation strength, and loss weighting. The ensemble config defines voting weights.

```bash
python training/train.py --config training/configs/ensemble.yaml
python training/evaluate.py --fairness-audit
python training/export.py --format onnx
```

---

## 5. Agentic Layer

### 5.1 Orchestrator (`src/agents/orchestrator.py`)

The main agent loop coordinates all downstream agents and tools. It receives the ensemble output + xAI summary and routes tasks to the appropriate sub-agents.

### 5.2 Clinical Summarizer (`src/agents/clinical_summarizer.py`)

The core intelligence of SkinSure. Combines three input streams:

- **Ensemble predictions** — top-k classes with confidence scores and inter-model agreement
- **Patient metadata** — age, sex, Fitzpatrick type, region, season, symptom duration, history
- **xAI heatmap summary** — spatial attention description, boundary alignment, artifact flags

A deterministic **context builder** assembles these into a structured prompt. The **LLM agent** (Claude via Anthropic SDK) then produces a `ClinicalSummary` in validated JSON:

```python
class ClinicalSummary(BaseModel):
    primary_condition: str          # "Eczema (Atopic dermatitis)"
    confidence_level: str           # "high" | "moderate" | "low"
    model_agreement: bool
    patient_summary: str            # 2–3 sentence plain-language explanation
    severity: str                   # "mild" | "moderate" | "urgent"
    urgency_reasoning: str
    contributing_factors: list[str] # ["monsoon season", "history of diabetes"]
    differential_notes: str
    recommended_action: str         # "self_care" | "schedule_dermatologist" | "immediate_referral"
    action_details: str
    confidence_caveats: list[str]
    recapture_needed: bool
```

**Agent rules (enforced via system prompt):**
- Never state a diagnosis as certain — use "the screening suggests" or "consistent with"
- Flag uncertainty explicitly when model agreement < 70%
- Factor in regional/seasonal epidemiology
- Downgrade confidence and request re-capture if xAI flags artifacts
- Never suggest self-care for urgent cases

### 5.3 Severity Triage (`src/agents/severity_triage.py`)

The LLM proposes a severity level, but hard deterministic rules in `severity_triage.py` override it:

- Melanoma prediction at any moderate+ confidence → always `urgent`
- Model confidence < 40% → block summary, request re-capture
- Rapidly spreading infection indicators → always `urgent`

This protects against the most dangerous LLM failure mode: hallucinating "mild" for a serious condition.

### 5.4 Downstream Agents

| Agent | File | Role |
|---|---|---|
| Treatment Advisor | `treatment_advisor.py` | OTC self-care recommendations |
| Doctor Referral | `doctor_referral.py` | Specialist matching by location and urgency |
| Pharmacy Locator | `pharmacy_locator.py` | Nearby pharmacy + medication availability |

### 5.5 Tools (`src/agents/tools/`)

| Tool | Purpose |
|---|---|
| `model_inference.py` | Run ensemble prediction |
| `explain_prediction.py` | Generate xAI heatmap |
| `search_doctors.py` | Query doctor directory API |
| `search_pharmacy.py` | Query pharmacy locator API |
| `patient_history.py` | Read/write patient records |

---

## 6. Explainability

Three complementary xAI methods ensure the system's reasoning is auditable (`src/explainability/`):

- **Grad-CAM / Grad-CAM++** (`gradcam.py`) — Highlights which regions of the image most influenced the prediction. Converted from raw heatmap to a structured spatial description (focus, boundary alignment, artifact concern) before being passed to the summarizer agent.
- **SHAP** (`shap_explainer.py`) — Feature-level attribution showing which image patches contributed to or against each class prediction.
- **LIME** (`lime_explainer.py`) — Superpixel-based local explanations; useful for validating that the model attends to the lesion rather than surrounding skin or image artifacts.

`report_generator.py` combines all three into a patient-facing explainability report.

---

## 7. API

Built with **FastAPI** (`api/`). All endpoints require JWT or API key authentication.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Image + metadata → `ClinicalSummary` |
| `POST` | `/explain` | Prediction → xAI heatmap report |
| `GET` | `/history` | Retrieve patient history |
| `POST` | `/history` | Save consultation record |
| `GET` | `/health` | Service health check |

Request/response schemas are defined as Pydantic models in `api/schemas/`.

---

## 8. Federated Learning

SkinSure uses federated learning (`src/federated/`) to improve models from hospital partner data without requiring raw images to leave hospital premises.

- **`client.py`** — Runs local training on partner hardware; sends only model weight updates
- **`server.py`** — Aggregates updates using FedAvg (or pluggable alternatives)
- **`privacy.py`** — Differential privacy utilities to bound information leakage per update
- **`communication.py`** — Encrypted transport for model updates

This is critical for healthcare compliance (DPDP Act, hospital data governance policies) and enables continuous model improvement across geographically distributed partners.

---

## 9. Project Structure

```
skinsure/
├── .claude/                  # Claude Code config & custom slash commands
├── CLAUDE.md                 # Global project context for Claude Code
├── docs/                     # Architecture, model card, API reference, domain knowledge
├── data/                     # Raw/processed datasets, metadata schemas, split manifests
├── src/
│   ├── preprocessing/        # Image cleaning, normalization, skin tone gating
│   ├── models/               # EfficientNet, ConvNeXt, Swin, ensemble, losses, metrics
│   ├── explainability/       # Grad-CAM, SHAP, LIME, report generator
│   ├── agents/               # Orchestrator, clinical summarizer, triage, referral, tools
│   ├── federated/            # FL client, server, privacy, communication
│   └── utils/                # Config, logging, image I/O, validators
├── api/                      # FastAPI app, routes, middleware, schemas
├── app/                      # Frontend (React Native / Flutter / Next.js — planned)
├── training/                 # Training scripts, evaluation, export, configs
├── tests/
│   ├── unit/                 # Preprocessing, model, agent, explainability tests
│   ├── integration/          # End-to-end pipeline & API tests
│   └── fairness/             # Per-skin-tone accuracy audit
├── scripts/                  # Dataset download, environment setup, batch preprocessing
├── configs/                  # App, agent, and preprocessing configs
├── docker/                   # Dockerfiles for API and training; docker-compose for local dev
└── .github/workflows/        # CI (lint + test) and automated model evaluation
```

---

## 10. Tech Stack

| Layer | Technology |
|---|---|
| ML Frameworks | PyTorch, timm |
| Model Backbones | EfficientNet, ConvNeXt, Swin Transformer |
| xAI | Grad-CAM, SHAP, LIME |
| LLM Agent | Anthropic Claude (via Anthropic SDK) |
| API | FastAPI + Pydantic |
| Data Validation | Pydantic v2 |
| Federated Learning | Custom (FedAvg) + OpenDP for differential privacy |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Config Management | YAML + Pydantic Settings |
| Package Management | uv / poetry (`pyproject.toml`) |
| Frontend (planned) | React Native / Flutter / Next.js |

---

## 11. Getting Started

```bash
# Clone and set up environment
git clone https://github.com/your-org/skinsure.git
cd skinsure
bash scripts/setup_env.sh

# Copy and configure environment variables
cp .env.example .env

# Download datasets
bash scripts/download_datasets.sh

# Run preprocessing
python scripts/run_preprocessing.py

# Start local dev stack
docker-compose -f docker/docker-compose.yaml up

# Run training
python training/train.py --config training/configs/ensemble.yaml

# Run evaluation with fairness audit
python training/evaluate.py --fairness-audit
```

**Claude Code custom commands:**

| Command | Action |
|---|---|
| `/preprocess` | Run the full preprocessing pipeline |
| `/train` | Launch a training run |
| `/evaluate` | Run the eval suite |
| `/deploy` | Build and deploy |

---

## 12. Testing & Fairness

```bash
# Unit tests
pytest tests/unit/

# Integration tests (end-to-end: image in → summary out)
pytest tests/integration/

# Fairness audit (per-Fitzpatrick-type accuracy)
pytest tests/fairness/test_skin_tone_bias.py
```

The fairness suite is also run automatically on every PR via GitHub Actions (`model-eval.yaml`). Any regression in per-tone accuracy fails the check.

---

## 13. Deployment

- **API service** — `docker/Dockerfile.api`
- **Training environment** (GPU) — `docker/Dockerfile.training`
- **Local dev stack** — `docker/docker-compose.yaml`

See `docs/deployment.md` for full infrastructure and deployment guide.

---

## Key Design Principles

**1. The LLM interprets; hard rules decide.**
Severity classification is hybrid — the agent proposes, deterministic Python overrides. This prevents the model from hallucinating a safe outcome for an urgent case.

**2. Context builder is deterministic; summarizer is not.**
The context builder assembles structured prompts without interpretation. All clinical reasoning happens inside the LLM agent, with a constrained system prompt.

**3. Fairness is a first-class CI check.**
Skin-tone accuracy bias is tested automatically on every model update, not treated as an afterthought.

**4. Privacy by architecture.**
Federated learning means patient images never leave hospital premises. Only weight deltas, protected by differential privacy, are transmitted.

---

*For architecture deep-dives, see `docs/architecture.md`. For domain knowledge on Indian skin tones and regional conditions, see `docs/indian-skin-tones.md`.*
