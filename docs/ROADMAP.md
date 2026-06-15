# SkinSure — Learning Roadmap

> A phase-by-phase build plan for SkinSure, written for someone **new to ML and backend development**.
> Each phase lists what you'll build, what you need to learn first, and how you'll know it's done.
>
> **Scope decisions baked into this plan:**
> - The **ensemble model is already trained** on curated data — no dataset sourcing or training phase.
> - Phases 0–10 build and deploy the **core product** (ensemble + clinical summarizer + basic Grad-CAM + API).
> - Phases 11–14 are the **research-heavy extensions** — full xAI suite, federated learning, and differential privacy. They build on a deployed, working system, so do them *after* the product is live.
> - Final core deployment target is a **major cloud provider** (AWS / GCP / Azure) via Docker.
>
> Work through phases in order. Don't jump ahead — each phase assumes the previous one works. Tackle one phase per sitting; finish its "Definition of done" before moving on.

---

## How to read this document

Every phase has the same shape:

- **Goal** — the one sentence outcome.
- **Learn first** — concepts to understand *before* writing code (with what to look up).
- **Build** — the concrete files/tasks to implement.
- **Definition of done** — the checklist that proves the phase works.
- **Why this matters** — how the phase connects to the whole.

A ⭐ marks the single most important thing to get right in that phase.

---

## Current codebase status (audited 2026-06-10)

A check of what's actually on disk, because it differs from earlier notes:

**Real and usable:**
- **Model layer** — `src/models/ensemble.py` (130 lines), `efficientnet.py`, `convnext.py`, `swin.py` all have real code. The `SkinEnsemble` wires all three backbones and outputs `Predictions`. **This is the project's one finished asset.**
- **Config** — `src/utils/config.py` + `configs/agents.yaml` load the summarizer model settings via `get_settings()`. Works.
- Training/eval notebooks in `src/models/`.

**Empty files (exist, but 0 bytes — must be written from scratch):**
- `api/schemas/patient.py` and `api/schemas/predictions.py` — **empty.** (Earlier docs claimed these were done; they are not.) The model layer imports `Predictions` from `predictions.py`, so this is an urgent gap.
- `src/agents/clinical_summarizer.py` — **empty.** (Earlier docs claimed "fully implemented"; it is not.)
- All agent stubs: `severity_triage.py`, `treatment_advisor.py`, `doctor_referral.py`, `pharmacy_locator.py`, `orchestrator.py`.
- `src/preprocessing/artifact_removal.py`.
- `src/agents/tools/` (empty directory).

**Not created yet:**
- `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, `.dockerignore`, `docker-compose.yml`.
- `api/main.py` and all routes.
- All other preprocessing modules (`quality_check.py`, `color_normalization.py`, `hair_removal.py`).
- `tests/`, `scripts/`, `docker/`, `.github/`, `src/explainability/`, `src/federated/`.

> ⚠️ **Correction to earlier plan:** the schemas and clinical summarizer are *empty*, not implemented. Phase 3 is now build-from-scratch, and there's a new **Phase 0.5** to write the schemas first — the model layer literally imports from them.

---

## Phase map (the whole journey at a glance)

Status: ✅ done · 🟡 partial · ⬜ not started

| # | Phase | Status | What you end up with |
|---|-------|--------|----------------------|
| 0 | Project setup & foundations | ⬜ | A runnable Python project with dependencies, env vars, and a virtual environment |
| 0.5 | Pydantic schemas | ⬜ | `PatientMetadata` + `Predictions` / `XAISummary` / `ClinicalSummary` (model layer depends on these) |
| 1 | Load the trained model & predict | 🟡 | A Python script that loads the ensemble and predicts on one image (ensemble code ✅; loader/script ⬜) |
| 2 | Preprocessing pipeline | ⬜ | Images cleaned & normalized before they reach the model |
| 3 | Clinical summarizer (LLM layer) | ⬜ | A trained-model prediction turned into a patient-friendly clinical summary |
| 4 | Severity triage (the safety layer) | ⬜ | Deterministic rules that override the LLM on urgent cases |
| 5 | Explainability (basic Grad-CAM) | ⬜ | A heatmap showing *where* the model looked |
| 6 | The API (FastAPI) | ⬜ | HTTP endpoints: send an image, get a clinical summary back |
| 7 | Testing & fairness audit | ⬜ | Automated tests, including per-skin-tone accuracy checks |
| 8 | Containerization (Docker) | ⬜ | The whole app running inside a reproducible container |
| 9 | Deployment to the cloud | ⬜ | A live URL anyone can hit |
| 10 | Hardening & monitoring | ⬜ | Logging, error handling, secrets, basic observability |
| 11 | Full xAI suite (SHAP + LIME) | ⬜ | Feature- and superpixel-level explanations beyond Grad-CAM |
| 12 | Remaining agents & router | ⬜ | Treatment advisor, doctor referral, pharmacy locator wired through an orchestrator |
| 13 | Federated learning | ⬜ | Hospitals train locally; only weight deltas reach the server |
| 14 | Differential privacy | ⬜ | Privacy-protected deltas for DPDP Act compliance |

> **Phases 0–10 = a deployable product. Phases 11–14 = research extensions** on top of it. Each later phase still assumes everything before it works.
>
> **Where to start:** Phase 0 → Phase 0.5 (schemas, unblocks everything) → Phase 1.

---

## Phase 0 — Project setup & foundations &nbsp;⬜ not started

**Goal:** Turn this folder into a real, runnable Python project you can install and import from.

### Learn first
- **Virtual environments** — why you isolate dependencies per project. Look up: `python -m venv`.
- **`pyproject.toml`** — the modern way to declare project metadata and dependencies. Look up: "PEP 621 pyproject.toml" and the `uv` or `pip` tooling.
- **Environment variables & secrets** — why API keys never go in code. Look up: `.env` files and `python-dotenv` / `pydantic-settings`.
- **Imports & packages** — why `from src.models.ensemble import SkinEnsemble` works only when the project is installed. Look up: "editable install `pip install -e .`".

### Build
1. Create `pyproject.toml` declaring the project and dependencies:
   - Runtime: `torch`, `timm`, `pillow`, `numpy`, `opencv-python`, `anthropic`, `pydantic`, `pydantic-settings`, `pyyaml`, `fastapi`, `uvicorn`, `python-multipart`.
   - Dev: `pytest`, `ruff` (linter/formatter).
2. Create a virtual environment and install the project in editable mode (`pip install -e .`).
3. Create `.env.example` documenting required keys — at minimum `ANTHROPIC_API_KEY`. Create your real `.env` (and confirm it's in `.gitignore`).
4. Confirm `configs/agents.yaml` loads via the existing `src/utils/config.py` → `get_settings()`.

### Definition of done
- [ ] `python -c "from src.utils.config import get_settings; print(get_settings())"` prints settings.
- [ ] `.env` exists locally, is gitignored, and `.env.example` is committed.
- [ ] `ruff check .` runs (warnings are fine for now).

> Note: `from src.models.ensemble import SkinEnsemble` will **fail until Phase 0.5**, because `ensemble.py` imports `Predictions` from the (currently empty) `api/schemas/predictions.py`. That import test is a DoD item for Phase 0.5, not here.

⭐ **Get the editable install working.** Everything else imports from `src/` and `api/` — if imports don't resolve, nothing downstream runs.

---

## Phase 0.5 — Pydantic schemas (the data contracts) &nbsp;⬜ not started

**Goal:** Write the Pydantic models the whole system passes around. **This is currently blocking everything** — `api/schemas/patient.py` and `api/schemas/predictions.py` are empty (0 bytes), and `src/models/ensemble.py` already imports `Predictions` from the empty file, so the model layer can't even be imported yet.

### Learn first
- **Pydantic v2 basics** — `BaseModel`, field types, defaults. Look up: Pydantic v2 "models".
- **Validators** — `@field_validator` and `@model_validator` to enforce rules at the data boundary. Look up: Pydantic v2 validators.
- **Enums for fixed sets** — Fitzpatrick types, severity levels, the 5 disease classes.

### Build
1. `api/schemas/patient.py` — `PatientMetadata` with the **Fitzpatrick IV–VI gate**: a validator that rejects skin types I–III with a clear error *before* anything reaches the model.
2. `api/schemas/predictions.py`:
   - `Predictions` — per-class probabilities, top class, `agreement_score`, and a `confidence_tier()` method mapping `agreement_score` → `"high"/"moderate"/"low"` (≥0.80 / ≥0.70 / <0.70). **Must match what `ensemble.py` constructs** — read `ensemble.py` to see exactly which fields it sets.
   - `XAISummary` — focus location, boundary alignment, `artifact_flag` (consumed by the summarizer in Phase 3).
   - `ClinicalSummary` — with a `@model_validator` enforcing `urgent → immediate_referral` at the schema level.

### Definition of done
- [ ] `python -c "from src.models.ensemble import SkinEnsemble; print('ok')"` now succeeds (the Phase 0 import that was blocked).
- [ ] A Fitzpatrick I–III `PatientMetadata` raises a validation error; IV–VI passes.
- [ ] `Predictions.confidence_tier()` returns sensible tiers; the fields line up with what `ensemble.py` builds.
- [ ] Constructing a `ClinicalSummary` with `urgent` severity but no immediate referral raises.

⭐ **Match `Predictions` to what `ensemble.py` already constructs.** The ensemble code is the finished, authoritative side of this contract — the schema must fit it, not the other way around. Read `ensemble.py` first.

---

## Phase 1 — Load the trained model & predict &nbsp;🟡 partial (ensemble code ✅ · loader + script ⬜)

**Goal:** Load the three trained backbone checkpoints into the `SkinEnsemble` and get a prediction on a single test image.

> Status: the `SkinEnsemble` class and the three backbone modules are **already written** (`src/models/`). What's missing is (a) the schemas it depends on (Phase 0.5) and (b) a script that loads checkpoints, transforms an image, and runs it. Requires Phase 0.5 first.

### Learn first
- **What a PyTorch checkpoint is** — a saved `state_dict` of weights. Look up: `torch.load`, `model.load_state_dict`, `model.eval()`.
- **Tensors & image-to-tensor** — how a JPG becomes a number grid the model accepts. Look up: `torchvision.transforms`, normalization (mean/std), `unsqueeze` for batch dimension.
- **CPU vs GPU (`device`)** — `torch.device("cuda" if available else "cpu")`.
- **Inference mode** — `with torch.no_grad():` and why it matters.

### Build
1. Locate your trained checkpoint files (one per backbone: EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny). Note their paths.
2. Write a small script `scripts/predict_one.py` that:
   - Builds `CheckpointPaths` and instantiates `SkinEnsemble`.
   - Loads & transforms one local image to the input size the models expect.
   - Runs inference and prints the `Predictions` object (top class, probabilities, `agreement_score`, confidence tier).
3. Confirm `Predictions.confidence_tier()` returns sensible high/moderate/low values.

### Definition of done
- [ ] Running `python scripts/predict_one.py path/to/image.jpg` prints a class label, probabilities, and `agreement_score`.
- [ ] You understand what `agreement_score` means (fraction of the 3 models agreeing with the ensemble).
- [ ] No errors about mismatched checkpoint keys or shapes.

⭐ **The transform must match how the models were trained** (same input size, same normalization). A mismatch produces garbage predictions that *look* valid — this is the #1 silent bug for beginners.

---

## Phase 2 — Preprocessing pipeline  ⬜ not started

**Goal:** Clean images before they reach the model so predictions are reliable.

### Learn first
- **Why preprocessing matters in dermatology** — hair, ruler marks, and lighting fool models. Skim `docs/indian_skin_tones.md`.
- **OpenCV basics** — reading images, color spaces (BGR vs RGB), blur detection. Look up: `cv2`, Laplacian variance for blur.
- **Color normalization** — making skin tone consistent across lighting. Look up: "Shades of Gray color constancy".

### Build (in order of value — stop when you have enough)
1. `src/preprocessing/quality_check.py` — reject blurry / too-small images (Laplacian variance threshold + min resolution).
2. `src/preprocessing/color_normalization.py` — Shades-of-Gray normalization (simplest effective method).
3. `src/preprocessing/hair_removal.py` — DullRazor algorithm + inpainting (optional but high-value for body images).
4. `src/preprocessing/artifact_removal.py` — already stubbed; fill in ruler/ink/bubble removal (optional, lower priority).
5. Wire these into a single `preprocess(image) -> image` function that Phase 1's predict path calls first.

### Definition of done
- [ ] A blurry image is rejected with a clear reason before hitting the model.
- [ ] A normalized image visibly evens out lighting (save before/after to compare).
- [ ] The predict script now runs preprocessing first, end to end.

⭐ **Quality check first.** Rejecting a bad image is more valuable than perfectly cleaning a good one — and it protects every downstream stage.

> Note: The **Fitzpatrick IV–VI gate** is *already enforced* in `api/schemas/patient.py` (`PatientMetadata` rejects skin types I–III). You don't build a `skin_tone_classifier` for this plan — the metadata gate covers it.

---

## Phase 3 — Clinical summarizer (the LLM layer) &nbsp;⬜ not started

**Goal:** Turn a raw `Predictions` object into a patient-friendly `ClinicalSummary` using Claude.

> ⚠️ `src/agents/clinical_summarizer.py` is **empty (0 bytes)** — you're building this from scratch, not just reviewing it. (Earlier notes said it was implemented; that was wrong.) It depends on the Phase 0.5 schemas. The config in `configs/agents.yaml` (model `claude-haiku-4-5-20251001`) is real and ready to use.

### Learn first
- **What an LLM API call looks like** — system prompt, user message, JSON response. Look up: the `anthropic` async SDK; read `/claude-api` references via the skill.
- **Async Python** — `async def` / `await`. Look up: `asyncio.run()`.
- **The two-phase design you'll build** — keep `build_context()` as pure Python (deterministic formatting) and `summarize()` as the async LLM call. Don't let interpretation leak into the context builder.
- **The post-processing rules** — artifact flag downgrades confidence; low agreement appends a caveat. These live in Python, *not* in the LLM.

### Build
1. Implement `build_context(predictions, patient, xai_summary) -> str` — deterministic formatting only, no interpretation.
2. Implement `async summarize(...) -> ClinicalSummary` — call Claude (model from `get_settings()`), instruct it to return **JSON only**, parse into `ClinicalSummary` (raise `ValueError` with the raw response on parse failure).
3. Add post-processing after the LLM response: if `xai_summary.artifact_flag` → downgrade `confidence_level` one tier and set `recapture_needed = True`; if `agreement_score < 0.70` and no caveat mentions "model disagreement" → append the standard caveat.
4. Write `scripts/summarize_one.py`: feed a `Predictions` + `PatientMetadata`, print the `ClinicalSummary`.

### Definition of done
- [ ] `scripts/summarize_one.py` produces a real `ClinicalSummary` from a prediction + metadata.
- [ ] The artifact-flag downgrade and low-agreement caveat both trigger (test each).
- [ ] `build_context()` contains zero interpretation — only formatting.

⭐ **Internalize "LLM interprets; Python decides."** This principle governs the whole agentic layer and is the foundation for Phase 4.

---

## Phase 4 — Severity triage (the safety layer)  ⬜ not started

**Goal:** Add deterministic rules that **override the LLM** on dangerous cases. This is the most safety-critical code in the project.

### Learn first
- **Defense in depth** — why a safety rule lives in *both* the LLM prompt and in Python.
- **The hard-override rules** (from CLAUDE.md):
  - Melanoma at moderate+ confidence → `urgent`.
  - Model confidence < 40% → block, request re-capture.
  - Rapidly spreading infection → `urgent`.

### Build
1. Implement `src/agents/severity_triage.py`: a pure function `triage(predictions, summary) -> final_severity` that applies the hard overrides *after* the LLM has proposed a severity.
2. The function must be able to **escalate** (LLM said non-urgent, rule says urgent) and **block** (low confidence → recapture).
3. Write unit tests for every override branch (these are non-negotiable — see Phase 7).

### Definition of done
- [ ] A melanoma prediction at moderate confidence comes out `urgent` even if the LLM hedged.
- [ ] A <40% confidence prediction is blocked with a recapture request.
- [ ] Tests cover each override path.

⭐ **The LLM never has the final word on urgent/non-urgent.** If you remember one rule from this whole project, it's this one.

---

## Phase 5 — Explainability (basic Grad-CAM)  ⬜ not started

**Goal:** Produce a heatmap showing which part of the image drove the prediction, and convert it into the `XAISummary` the summarizer consumes.

### Learn first
- **What Grad-CAM is** — gradients flowing into the last conv layer reveal "where the model looked". Look up: "Grad-CAM intuition" (the original paper's figure is enough).
- **Hooks in PyTorch** — how to capture activations/gradients from a layer. Look up: `register_forward_hook`.
- **A library shortcut** — `pytorch-grad-cam` (pip package) saves you writing hooks by hand. Strongly recommended for a beginner.

### Build
1. Create `src/explainability/gradcam.py`: given a model + image, return a heatmap array.
2. Convert the heatmap into an `XAISummary` (focus location, boundary alignment, artifact flag) — this is the bridge to the summarizer.
3. Overlay the heatmap on the original image and save it for visual inspection.

### Definition of done
- [ ] You can produce a heatmap PNG that visibly highlights the lesion, not the background.
- [ ] The heatmap is converted into a valid `XAISummary`.
- [ ] If the heatmap focuses on an artifact (e.g. ruler), `artifact_flag` is set — and you've watched it downgrade confidence in Phase 3's post-processing.

⭐ **Validate the model looks at the lesion, not the background.** A model that's right for the wrong reason is a model you can't trust clinically.

---

## Phase 6 — The API (FastAPI)  ⬜ not started

**Goal:** Wrap everything in HTTP endpoints so an app could send an image and receive a clinical summary.

### Learn first
- **What a web API is** — request → handler → response. Look up: FastAPI "first steps" tutorial.
- **File uploads** — receiving an image over HTTP. Look up: FastAPI `UploadFile`, `python-multipart`.
- **Pydantic as request/response models** — your existing schemas *are* your API contract. Look up: FastAPI + Pydantic response models.
- **Running a server** — `uvicorn api.main:app --reload`.

### Build
1. Create `api/main.py` with the FastAPI app instance.
2. Implement endpoints (start with the first two):
   - `POST /predict` — image + `PatientMetadata` → run preprocessing → ensemble → triage → summarizer → return `ClinicalSummary`. ⭐
   - `GET /health` — returns `{"status": "ok"}` (you'll need this for deployment health checks).
   - `POST /explain` — prediction → Grad-CAM report (Phase 5).
   - `GET /history`, `POST /history` — start with in-memory or a simple SQLite store; a real DB can come later.
3. Add an orchestrator (`src/agents/orchestrator.py`) that sequences preprocessing → model → triage → summarizer so `main.py` stays thin.

### Definition of done
- [ ] `uvicorn api.main:app --reload` starts without errors.
- [ ] Visiting `/docs` (FastAPI's auto Swagger UI) shows your endpoints.
- [ ] Uploading a test image to `POST /predict` returns a real `ClinicalSummary` JSON.
- [ ] `GET /health` returns ok.

⭐ **`POST /predict` is the whole product in one endpoint.** Get the full chain — image in, clinical summary out — working before polishing the others.

---

## Phase 7 — Testing & fairness audit  ⬜ not started

**Goal:** Lock in correctness with automated tests, including the project's signature **per-skin-tone fairness check**.

### Learn first
- **pytest basics** — test functions, `assert`, fixtures. Look up: pytest "getting started".
- **Why fairness is a first-class test here** — a model that's accurate on lighter tones but not Fitzpatrick IV–VI is a failure, by design of this project.
- **Mocking the LLM** — you don't want to call (and pay for) Claude in every test. Look up: `unittest.mock` / `pytest` monkeypatch.

### Build
1. `tests/unit/` — preprocessing, severity triage (all override branches), clinical summarizer (with the LLM mocked), explainability.
2. `tests/integration/` — end-to-end: image in → `ClinicalSummary` out, hitting the real chain (LLM mocked).
3. `tests/fairness/test_skin_tone_bias.py` — per-Fitzpatrick-subtype accuracy on a held-out set; fail if any subtype regresses.

### Definition of done
- [ ] `pytest tests/` passes.
- [ ] Severity-triage overrides are all covered (the most important tests in the suite).
- [ ] The fairness test runs and reports per-subtype accuracy.

⭐ **Severity-triage and fairness tests are the two you must never skip.** They encode the project's safety and equity promises.

---

## Phase 8 — Containerization (Docker)  ⬜ not started

**Goal:** Package the app so it runs identically on your machine and in the cloud.

### Learn first
- **What Docker is** — a reproducible box holding your app + dependencies. Look up: Docker "getting started", `Dockerfile`, `docker build`, `docker run`.
- **Why containers for deployment** — "works on my machine" → "works everywhere".
- **Image size & layers** — model weights are large; learn `.dockerignore` and multi-stage builds.
- **docker-compose** — running the app (and later a DB) together. Look up: `docker-compose.yml`.

### Build
1. `docker/Dockerfile` — base Python image, install deps, copy code + model weights, run `uvicorn`.
2. `.dockerignore` — exclude `.env`, `.git`, notebooks, caches.
3. `docker-compose.yml` — at least the API service; add a DB service if you went beyond in-memory history.
4. Decide how model weights ship: baked into the image vs. mounted/downloaded at startup (large weights often shouldn't live in the image).

### Definition of done
- [ ] `docker build` produces an image.
- [ ] `docker run` (with `ANTHROPIC_API_KEY` passed in) serves the API, and `GET /health` responds.
- [ ] `POST /predict` works against the containerized app.

⭐ **Pass secrets at runtime, never bake them into the image.** A committed/baked API key is a security incident.

---

## Phase 9 — Deployment to the cloud  ⬜ not started

**Goal:** Get a live URL on a major cloud provider (AWS / GCP / Azure).

### Learn first
- **Container hosting options** — the beginner-friendly managed ones:
  - **GCP Cloud Run** (recommended starting point — serverless containers, scales to zero, simplest).
  - **AWS App Runner / ECS Fargate**.
  - **Azure Container Apps**.
- **Container registries** — where your image lives before deploy (Artifact Registry / ECR / ACR).
- **Secrets in the cloud** — Secret Manager (GCP) / Secrets Manager (AWS) / Key Vault (Azure) for `ANTHROPIC_API_KEY`.
- **Health checks & cold starts** — why `GET /health` matters; loading a big model on cold start is slow.

### Build
1. Push your Docker image to the cloud registry.
2. Deploy the container (start with **Cloud Run** if undecided — least to learn).
3. Wire `ANTHROPIC_API_KEY` via the cloud secret manager, not env files.
4. Configure the health check to hit `/health`.
5. Set memory/CPU high enough to hold the ensemble in RAM (model loading is memory-hungry — undersizing causes silent OOM crashes).

### Definition of done
- [ ] A public (or auth-gated) URL responds to `GET /health`.
- [ ] `POST /predict` returns a `ClinicalSummary` from the deployed service.
- [ ] The API key is stored in the cloud secret manager, not in the image or env files.

⭐ **Right-size memory for the model.** Beginners deploy, hit out-of-memory on the first real prediction, and get confusing crashes. Provision RAM for all three backbones up front.

---

## Phase 10 — Hardening & monitoring (production readiness)  ⬜ not started

**Goal:** Make the deployed service safe, observable, and maintainable.

### Learn first
- **Structured logging** — request IDs, timing, errors. Look up: Python `logging`, JSON logs.
- **Error handling** — never leak stack traces to users; return clean error responses.
- **Rate limiting & auth** — protect the endpoint and your LLM bill. Look up: API keys / simple token auth, rate limiting middleware.
- **Cost & monitoring** — track Anthropic API spend and cloud usage. Look up: your provider's monitoring dashboard.
- **CI/CD** — auto-run tests on every push, auto-deploy on green. Look up: GitHub Actions (`.github/workflows/`).

### Build
1. Add structured logging through the request lifecycle.
2. Add an exception handler so failures return clean JSON, not tracebacks.
3. Add basic auth / an API key to protect `POST /predict`.
4. Add a GitHub Actions workflow: run `pytest` (including the **fairness test**) on every PR; deploy on merge to `main`.
5. Set up basic monitoring/alerts (uptime, error rate, LLM spend).

### Definition of done
- [ ] Errors return clean responses; logs are structured and searchable.
- [ ] `POST /predict` requires auth.
- [ ] CI runs the full test suite (fairness included) on every PR — a fairness regression blocks merge.
- [ ] You can see request volume, errors, and LLM cost on a dashboard.

⭐ **CI must run the fairness suite on every PR.** This is the mechanism that keeps the equity promise true as the code evolves — it's in the project's core design invariants.

---

## Phase 11 — Full xAI suite (SHAP + LIME)  ⬜ not started

**Goal:** Go beyond Grad-CAM's "where" to feature- and superpixel-level explanations of *why* the model predicted a class, and use them to validate the model attends to the lesion, not the background.

### Learn first
- **The three xAI methods and what each answers:**
  - **Grad-CAM** (Phase 5) — *spatial*: which region of the image mattered.
  - **SHAP** — *feature attribution*: how much each input feature/patch pushed the score up or down. Look up: "SHAP values intuition", `shap.DeepExplainer` / `GradientExplainer` for image models.
  - **LIME** — *local surrogate*: perturb superpixels, fit a simple model locally to see which superpixels drove the prediction. Look up: `lime.lime_image`, superpixel segmentation (SLIC).
- **Why three, not one** — they cross-check each other; agreement across methods is a trust signal, disagreement a red flag.
- **Cost & latency** — SHAP and LIME are *slow* (many forward passes). They're for offline analysis / an `/explain` deep-dive, not the hot `/predict` path.

### Build
1. `src/explainability/shap_explainer.py` — patch-level attribution for a prediction; save an attribution overlay.
2. `src/explainability/lime_explainer.py` — superpixel explanation; highlight the superpixels supporting the predicted class.
3. Extend the `XAISummary` bridge (from Phase 5) so artifact detection can draw on all three methods, not just Grad-CAM.
4. Upgrade `POST /explain` to optionally return SHAP/LIME reports (gated behind a flag so it's never on the latency-sensitive `/predict` path).

### Definition of done
- [ ] For one test image you can produce Grad-CAM, SHAP, and LIME overlays side by side.
- [ ] All three broadly agree the model is looking at the lesion (or you've found a case where they don't, and understand why).
- [ ] `POST /explain` can return the richer report; `POST /predict` latency is unchanged.

⭐ **Use the three methods to cross-validate, not just decorate.** Their *agreement* is the actual deliverable — a clinical trust signal, not a pretty picture.

---

## Phase 12 — Remaining agents & router  ⬜ not started

**Goal:** Complete the agentic layer so a clinical summary flows into actionable next steps — treatment guidance, referral, and pharmacy lookup — coordinated by an orchestrator/router.

### Learn first
- **The agent topology** (from CLAUDE.md's architecture diagram): orchestrator → clinical summarizer + severity triage + treatment advisor → router → pharmacy locator / doctor referral.
- **Tool use with the LLM** — how an agent calls a function (a "tool") and feeds the result back. Look up: Anthropic tool use; read `/claude-api` references.
- **The "LLM interprets; Python decides" boundary again** — referral *urgency* is decided by Phase 4's triage, not by the advisor agent.

### Build
1. `src/agents/treatment_advisor.py` — given a `ClinicalSummary`, propose OTC/self-care guidance for non-urgent cases (with hard guardrails: never prescribe; always defer to a clinician for urgent/uncertain cases).
2. `src/agents/doctor_referral.py` — for urgent/immediate-referral cases, produce a referral recommendation (specialty, urgency, what to tell the doctor).
3. `src/agents/pharmacy_locator.py` — a tool that, given a location, returns nearby pharmacies (start with a mock/static data source; a real maps API is optional).
4. `src/agents/tools/` — implement the tool functions the agents call.
5. Flesh out `src/agents/orchestrator.py` into a real **router**: severity decides whether to go to treatment advice (non-urgent) or doctor referral (urgent).

### Definition of done
- [ ] A non-urgent case flows: summary → treatment advice → (optional) pharmacy lookup.
- [ ] An urgent case flows: summary → triage escalation → doctor referral, and treatment advice is *suppressed*.
- [ ] The orchestrator's routing is covered by tests (urgent vs. non-urgent paths).

⭐ **Urgent cases must never receive self-treatment advice.** The router suppresses the treatment advisor whenever triage says urgent — verify this with a test.

---

## Phase 13 — Federated learning  ⬜ not started

**Goal:** Let hospital partners improve the model on their own local data without that data ever leaving their premises — only model weight *deltas* are sent to a central server.

### Learn first
- **Why federated, not centralized** — Indian patient data can't be pooled centrally (privacy + DPDP Act). The model goes to the data, not the data to the model. Look up: "federated learning intuition", the FedAvg paper.
- **FedAvg** — each client trains a few local epochs; the server averages their weight updates into a new global model; repeat. Look up: federated averaging algorithm.
- **What actually crosses the network** — weight deltas (or full local weights), *never* raw images.
- **Client/server split** — clients hold data + do training; the server only aggregates.

### Build
1. `src/federated/client.py` — load the global model, train locally for N epochs on local data, compute and return weight deltas.
2. `src/federated/server.py` — FedAvg aggregation: collect deltas from clients, average, broadcast the updated global model.
3. `src/federated/communication.py` — the transport layer that moves deltas between client and server (start simple: local simulation of N clients before any real networking).
4. Simulate a federated round end to end with 2–3 mock clients on partitioned data, and confirm the global model improves.

### Definition of done
- [ ] You can run one simulated FedAvg round across ≥2 clients and produce an updated global model.
- [ ] You can confirm **no raw image data** is ever transmitted — only weights/deltas.
- [ ] The federated-trained model loads back into the `SkinEnsemble` and predicts.

⭐ **Raw data never leaves the client.** If an image (or anything reconstructable into one) crosses the wire, the whole point — and the compliance basis — is gone.

---

## Phase 14 — Differential privacy  ⬜ not started

**Goal:** Add mathematical privacy guarantees to the weight deltas, so even the deltas can't be reverse-engineered to leak information about an individual patient — the compliance capstone.

### Learn first
- **Why deltas alone aren't enough** — model updates can leak training data (membership/reconstruction attacks). DP adds calibrated noise so no single patient measurably changes the output. Look up: "differential privacy intuition", the (ε, δ) privacy budget.
- **The privacy budget (ε)** — smaller ε = more privacy, more noise, lower accuracy. You're tuning a privacy/utility trade-off.
- **DP-SGD & gradient clipping** — clip per-example gradients, add Gaussian noise. Look up: DP-SGD; the **OpenDP** library and Opacus for PyTorch.
- **DPDP Act context** — why this matters for deploying with Indian hospitals. Skim the compliance angle in `docs/architecture.md`.

### Build
1. `src/federated/privacy.py` — wrap the Phase 13 client training with DP: per-example gradient clipping + calibrated noise (via OpenDP / Opacus), and track the spent privacy budget.
2. Make ε a configurable knob and measure the accuracy cost at a few ε values (the privacy/utility curve).
3. Integrate DP into the federated round from Phase 13 so deltas are privatized *before* they leave the client.

### Definition of done
- [ ] Federated rounds run with DP enabled; you can report the (ε, δ) budget spent.
- [ ] You've measured and can explain the accuracy cost of your chosen ε.
- [ ] Privatization happens on the **client**, before any delta is transmitted.

⭐ **Noise is added on the client, before transmission — never on the server.** Privacy that depends on trusting the server isn't privacy.

---

## Still optional after Phase 14

Not research, just remaining product polish you can pick up anytime:

- **A real datastore & patient history** — Postgres + migrations instead of in-memory/SQLite.
- **A real pharmacy/maps integration** — replace the mock locator with a live API.

---

## A few principles to carry through every phase

1. **LLM interprets; Python decides.** Deterministic rules always override the model on safety.
2. **Fairness is a test, not an afterthought.** Per-Fitzpatrick accuracy is checked on every change.
3. **Reject bad input early.** Quality checks and the Fitzpatrick gate protect everything downstream.
4. **Secrets never touch git or images.** `.env` locally, secret manager in the cloud.
5. **Finish a phase's Definition of done before moving on.** Half-built phases compound into confusion.
