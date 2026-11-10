# SkinSure

Clinical decision-support API that screens photos of skin lesions for Fitzpatrick IV–VI (Indian) skin tones — built, tested, and shipped as a FastAPI service with a 3-model ensemble, Grad-CAM explainability, deterministic safety overrides, and an observability layer, not just a trained classifier.

## Demo


## Pipeline Stages

**1. Ingestion & gating**
- Fitzpatrick IV–VI type enforced at the Pydantic schema boundary (`api/schemas/patient.py`) — rejects out-of-scope skin tones before any model runs
- Quality gate rejects images under 200px short-side or near-featureless; blur threshold tuned per-class from a 1,500-image sample to avoid disproportionately rejecting BCC/Melanoma photos

**2. Preprocessing**
- Modular pipeline (hair removal, artifact inpainting, color normalization) with every pixel-changing step gated off by default, because the models were trained on resize + normalization only
- Each optional step is independently toggleable via config rather than hardcoded, so the pipeline can be re-tuned without touching inference code

**3. Core system — ensemble inference**
- Weighted-average ensemble of EfficientNet-B1, ConvNeXt-Tiny, and Swin-Tiny, each contributing a different inductive bias (texture / lighting invariance / global context)
- No fallback-checkpoint path by design — a missing model weight raises an error instead of silently degrading to a stale or mismatched generation

**4. Explainability & safety triage**
- Grad-CAM heatmaps per backbone, chosen over SHAP for the live path on latency (~5s vs. minutes on CPU for patch-based methods) and native spatial-map output
- Deterministic Python rules sit downstream of the LLM-generated clinical summary and can only escalate or block severity (e.g., melanoma at moderate+ confidence → forced "urgent") — the LLM never gets the final word on urgency

**5. Output / delivery**
- LLM (Claude Haiku) turns model output into a structured clinical summary, constrained by a Pydantic schema with a validator enforcing `urgent → immediate_referral` at the data layer as a second safety check
- Cost-accounting module prices every LLM call against a rate table, and any script that spends money is hard-gated behind an explicit env var + `--max-usd` flag with no default

**6. Observability & deployment**
- OpenTelemetry tracing is optional-by-design: the app boots and serves with the telemetry stack entirely absent, verified by import-blocking the dependency in tests
- PII redaction is allowlist-based (not blocklist) — patient name is structurally excluded from traces, patient ID is salted-HMAC'd, images reduce to hash + dimensions

## Architecture

FastAPI orchestrator (`run_screening`) chains quality gate → ensemble → Grad-CAM explainer → LLM summarizer → severity triage, emitting a decision span at each stage for audit.

`[architecture diagram placeholder — insert system diagram image here]`

## Tech Stack

- **Language:** Python 3.10+
- **API / serving:** FastAPI, Uvicorn, Pydantic v2
- **ML:** PyTorch, timm (EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny)
- **Explainability:** Grad-CAM (custom per-backbone hooks)
- **LLM integration:** Anthropic Claude (async SDK), prompt-based clinical reasoning with schema-validated output
- **Observability:** OpenTelemetry (optional extra), structured tracing, PII redaction
- **Config:** YAML + `pydantic-settings`
- **Testing:** pytest, 24 test files (unit, integration, fairness), network-blocked test runs, fake LLM clients
- **Image processing:** OpenCV, Pillow

## Limitations

- **Test-set contamination was found and corrected, not assumed absent.** The original evaluation set had ~22% near-duplicate overlap with training data, caught via perceptual hashing and removed before reporting metrics. That method catches near-identical duplicates, not subtler leakage (e.g. the same lesion photographed from a different angle), so residual leakage risk isn't fully ruled out — and results from the original, contaminated set and the cleaned set are never comparable to each other.
- **Explainability is heuristic and structurally unverifiable against ground truth.** The heatmap-to-region bridge maps Grad-CAM activations to a lesion focus/boundary call using brightness segmentation, not annotated lesion masks — and annotated masks are a deliberate non-goal, not a backlog item. So the explanation output is advisory evidence of where the model looked, never proof it looked at the lesion itself.
- **Safety-critical scope gating depends on self-reported patient input.** The Fitzpatrick IV–VI gate — the thing that makes the model's accuracy claims apply at all — relies on the patient truthfully reporting their skin type, with no image-based verification. Every downstream guarantee is conditional on that input being honest.

## Links

- **Repo:** this repository (branch `ijist-paper`)
- **Demo:** not yet deployed
