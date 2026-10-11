# SkinSure — Target Architecture

> Planned folder layout and import map. Many files here do not exist yet — `CLAUDE.md` ("Current Implementation State") is the source of truth for what is built, and `fedler.md` for federated learning. Files marked "exists" are implemented.

```
skinsure/
│
├── .claude/                          # Claude Code config
│   ├── settings.json                 # model preferences, permissions
│   └── commands/                     # custom slash commands
│       ├── preprocess.md             # /preprocess — run preprocessing pipeline
│       ├── train.md                  # /train — launch training run
│       ├── evaluate.md               # /evaluate — run eval suite
│       └── deploy.md                 # /deploy — build & deploy
│
├── CLAUDE.md                         # ★ Global context + source of truth for current state
│                                     #   project overview, conventions,
│                                     #   tech stack, coding standards
├── fedler.md                         # ★ Source of truth for federated learning & DP
├── QnA.md                            # learning log (not normative)
├── models/
│   └── checkpoints/                  # trained .pth files: efficientNet/, convNext/, swin/
│
├── docs/
│   ├── architecture.md               # system design & data flow
│   ├── ROADMAP.md                    # phase-by-phase build plan
│   ├── model-card.md                 # (planned) ensemble model card (bias, limitations)
│   ├── api-reference.md              # (planned) REST endpoint docs
│   ├── deployment.md                 # (planned) infra & deployment guide
│   └── indian_skin_tones.md          # domain knowledge: Fitzpatrick IV-VI,
│                                     #   common conditions, regional patterns
│
├── data/
│   ├── raw/                          # original datasets (gitignored)
│   ├── processed/                    # preprocessed images (gitignored)
│   ├── metadata/                     # patient metadata schemas & samples
│   │   └── schema.json
│   ├── splits/                       # train/val/test split manifests
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── label_map.json                # disease class → index mapping
│
├── src/
│   ├── preprocessing/
│   │   ├── CLAUDE.md                 # ★ Local context: preprocessing conventions
│   │   ├── __init__.py
│   │   ├── pipeline.py               # orchestrates all preprocessing steps
│   │   ├── inference_transform.py    # resize 224 + ImageNet normalisation (exists)
│   │   ├── color_normalization.py    # shades-of-gray, Macenko, Reinhard
│   │   ├── hair_removal.py           # DullRazor / inpainting
│   │   ├── augmentation.py           # tone-aware augmentations
│   │   ├── skin_tone_classifier.py   # FUTURE SCOPE — image-based Fitzpatrick detection;
│   │   │                             #   the IV–VI gate is enforced by PatientMetadata today
│   │   ├── quality_check.py          # blur detection, resolution validation
│   │   └── artifact_removal.py       # ruler marks, ink, bubbles
│   │
│   ├── models/
│   │   ├── CLAUDE.md                 # ★ Local context: model architecture notes
│   │   ├── __init__.py
│   │   ├── efficientnet.py           # EfficientNet backbone
│   │   ├── convnext.py               # ConvNeXt backbone
│   │   ├── swin.py                   # Swin Transformer backbone
│   │   ├── ensemble.py               # weighted ensemble / meta-learner
│   │   ├── losses.py                 # focal loss, class-balanced loss
│   │   └── metrics.py                # per-class accuracy, AUC, fairness metrics
│   │
│   ├── explainability/
│   │   ├── CLAUDE.md                 # ★ Local context: xAI approach
│   │   ├── __init__.py
│   │   ├── gradcam.py                # Grad-CAM / Grad-CAM++
│   │   ├── shap_explainer.py         # SHAP-based feature attribution
│   │   ├── lime_explainer.py         # LIME for image explanations
│   │   └── report_generator.py       # combines heatmaps into patient report
│   │                                 #   returns XAISummary (from api/schemas/predictions.py)
│   │
│   ├── agents/
│   │   ├── CLAUDE.md                 # ★ Local context: agent design patterns
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # main agent loop / router
│   │   ├── clinical_summarizer.py    # prediction + metadata → summary
│   │   │                             #   imports: Predictions, XAISummary, ClinicalSummary
│   │   │                             #   imports: PatientMetadata
│   │   │                             #   XAISummary no longer defined here
│   │   ├── severity_triage.py        # urgency classification agent
│   │   │                             #   imports: ClinicalSummary
│   │   ├── treatment_advisor.py      # next-steps & self-care suggestions
│   │   ├── doctor_referral.py        # specialist matching by location
│   │   ├── pharmacy_locator.py       # nearby pharmacy + med availability
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── model_inference.py    # tool: run ensemble prediction
│   │       │                         #   returns Predictions (from api/schemas/predictions.py)
│   │       ├── explain_prediction.py # tool: generate xAI heatmap
│   │       │                         #   returns XAISummary (from api/schemas/predictions.py)
│   │       ├── search_doctors.py     # tool: query doctor directory API
│   │       ├── search_pharmacy.py    # tool: query pharmacy locator API
│   │       └── patient_history.py    # tool: read/write patient records
│   │
│   ├── federated/                    # DEFERRED — design & phases: see fedler.md
│   │   ├── CLAUDE.md                 # ★ Local context: FL strategy
│   │   ├── __init__.py
│   │   ├── client.py                 # Flower NumPyClient representing ONE HOSPITAL (3 simulated)
│   │   ├── server.py                 # Flower FedAvg strategy / aggregation
│   │   ├── privacy.py                # (deferred, fedler Phase 5) Flower DP strategies; OpenDP only if required
│   │   └── communication.py          # Flower gRPC transport configuration (fedler Phase 3)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_utils.py            # train_one_epoch / evaluate (reused by federated clients; see fedler.md)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # pydantic settings / env loader
│       ├── logger.py                 # structured logging
│       ├── image_io.py               # image load/save helpers
│       └── validators.py             # input validation schemas
│
├── api/
│   ├── CLAUDE.md                     # ★ Local context: API conventions
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entrypoint
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py                # POST /predict — image → diagnosis
│   │   ├── explain.py                # POST /explain — prediction → heatmap
│   │   ├── history.py                # GET/POST patient history
│   │   └── health.py                 # GET /health — service health check
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                   # JWT / API key auth
│   │   └── rate_limit.py             # per-user rate limiting
│   └── schemas/
│       ├── __init__.py
│       ├── predictions.py            # full ML pipeline type contract
│       │                             #   Predictions      — ensemble output
│       │                             #     · primary_label, primary_confidence
│       │                             #     · secondary_label, secondary_confidence
│       │                             #     · agreement_score
│       │                             #     · confidence_tier() method
│       │                             #   XAISummary       — explainability output
│       │                             #     · focus_description
│       │                             #     · boundary_alignment
│       │                             #     · artifact_flag
│       │                             #   ClinicalSummary  — agent output
│       │                             #     · severity (routine/monitor/urgent), summary, recommended_action
│       │                             #     · confidence_level, confidence_caveats, recapture_needed
│       │                             #     · immediate_referral
│       │                             #     · @model_validator: urgent → immediate_referral
│       └── patient.py                # PatientMetadata model
│                                     #   · patient_id, patient_name, patient_age, patient_sex
│                                     #   · fitzpatrick_type (IV–VI gate)
│                                     #   · patient_location (region), current_season, body_area
│                                     #   · symptom_duration, medical_history
│                                     #   (symptom_duration is stored as symptom_duration_days)
│
├── web/                              # website frontend (future)
│   ├── CLAUDE.md                     # ★ Local context: frontend stack
│   └── ...                           # framework not yet chosen
│
├── training/
│   ├── CLAUDE.md                     # ★ Local context: training recipes
│   ├── train.py                      # main training script
│   ├── evaluate.py                   # evaluation + fairness audit
│   ├── export.py                     # model export (ONNX / TorchScript)
│   └── configs/
│       ├── efficientnet.yaml
│       ├── convnext.yaml
│       ├── swin.yaml
│       └── ensemble.yaml
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_models.py
│   │   ├── test_agents.py
│   │   └── test_explainability.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py      # image in → summary out
│   │   └── test_api.py
│   └── fairness/
│       └── test_skin_tone_bias.py    # per-tone accuracy audit
│
├── scripts/
│   ├── download_datasets.sh          # fetch ISIC / Dermnet / custom data
│   ├── setup_env.sh                  # environment setup
│   ├── predict_one.py                # single-image ensemble inference (exists)
│   ├── summarize_one.py              # single summarizer call (exists)
│   ├── make_federated_shards.py      # IID / non-IID hospital shard manifests (fedler Phase 1)
│   └── run_preprocessing.py          # batch preprocessing runner
│
├── configs/
│   ├── app.yaml                      # app-level config
│   ├── agents.yaml                   # summarizer model, max_tokens, temperature (exists)
│   ├── models.yaml                   # checkpoint paths + ensemble weights (exists)
│   └── preprocessing.yaml            # preprocessing pipeline params
│
├── docker/
│   ├── Dockerfile.api                # API service
│   ├── Dockerfile.training           # training environment (GPU)
│   └── docker-compose.yaml           # local dev stack
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                   # lint + test on PR
│       └── model-eval.yaml           # automated model evaluation
│
├── .env.example                      # environment variable template
├── .gitignore
├── pyproject.toml                    # project deps (uv / poetry)
├── Makefile                          # common commands shorthand
└── README.md
```

## Import map (what changed)

  api/schemas/predictions.py
    exports → Predictions, XAISummary ★, ClinicalSummary

  api/schemas/patient.py
    exports → PatientMetadata

  src/agents/tools/model_inference.py
    imports ← Predictions                       (from api/schemas/predictions)

  src/agents/tools/explain_prediction.py
    imports ← XAISummary ★                      (from api/schemas/predictions)
    returns → XAISummary instance

  src/agents/clinical_summarizer.py
    imports ← Predictions, XAISummary ★, ClinicalSummary  (from api/schemas/predictions)
    imports ← PatientMetadata                   (from api/schemas/patient)
    removed → XAISummary dataclass definition ★

  src/agents/severity_triage.py
    imports ← ClinicalSummary                   (from api/schemas/predictions)

  src/explainability/report_generator.py
    imports ← XAISummary ★                      (from api/schemas/predictions)