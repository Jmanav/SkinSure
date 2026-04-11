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
├── CLAUDE.md                         # ★ Global context for Claude Code
│                                     #   project overview, conventions,
│                                     #   tech stack, coding standards
│
├── docs/
│   ├── architecture.md               # system design & data flow
│   ├── model-card.md                 # ensemble model card (bias, limitations)
│   ├── api-reference.md              # REST/gRPC endpoint docs
│   ├── deployment.md                 # infra & deployment guide
│   └── indian-skin-tones.md          # domain knowledge: Fitzpatrick IV-VI,
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
│   │   ├── color_normalization.py    # shades-of-gray, Macenko, Reinhard
│   │   ├── hair_removal.py           # DullRazor / inpainting
│   │   ├── augmentation.py           # tone-aware augmentations
│   │   ├── skin_tone_classifier.py   # Fitzpatrick type detection (IV-VI gate)
│   │   ├── quality_check.py          # blur detection, resolution validation
│   │   └── artifact_removal.py       # ruler marks, ink, bubbles
│   │
│   ├── models/
│   │   ├── CLAUDE.md                 # ★ Local context: model architecture notes
│   │   ├── __init__.py
│   │   ├── efficientnet.py           # EfficientNet backbone
│   │   ├── convnext.py               # ConvNeXt backbone
│   │   ├── swin_transformer.py       # Swin Transformer backbone
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
│   │
│   ├── agents/
│   │   ├── CLAUDE.md                 # ★ Local context: agent design patterns
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # main agent loop / router
│   │   ├── clinical_summarizer.py    # prediction + metadata → summary
│   │   ├── severity_triage.py        # urgency classification agent
│   │   ├── treatment_advisor.py      # next-steps & self-care suggestions
│   │   ├── doctor_referral.py        # specialist matching by location
│   │   ├── pharmacy_locator.py       # nearby pharmacy + med availability
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── model_inference.py    # tool: run ensemble prediction
│   │       ├── explain_prediction.py # tool: generate xAI heatmap
│   │       ├── search_doctors.py     # tool: query doctor directory API
│   │       ├── search_pharmacy.py    # tool: query pharmacy locator API
│   │       └── patient_history.py    # tool: read/write patient records
│   │
│   ├── federated/
│   │   ├── CLAUDE.md                 # ★ Local context: FL strategy
│   │   ├── __init__.py
│   │   ├── client.py                 # local training client
│   │   ├── server.py                 # aggregation server (FedAvg etc.)
│   │   ├── privacy.py                # differential privacy utilities
│   │   └── communication.py          # secure model update transport
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
│       ├── prediction.py             # request/response models
│       └── patient.py                # patient metadata models
│
├── app/                              # frontend (future)
│   ├── CLAUDE.md                     # ★ Local context: frontend stack
│   └── ...                           # React Native / Flutter / Next.js
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
│   └── run_preprocessing.py          # batch preprocessing runner
│
├── configs/
│   ├── app.yaml                      # app-level config
│   ├── agents.yaml                   # agent prompts, tool configs
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