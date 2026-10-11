# SkinSure
SkinSure is an advanced AI-powered diagnostic tool designed to analyze skin disease images instantly. Tailored specifically for Indian skin tones, SkinSure utilizes proprietary image preprocessing and an ensemble model to deliver high-accuracy predictions, ensuring accessibility and early detection of skin conditions.

> Work in progress — a clinical decision-support prototype for Fitzpatrick IV–VI skin, not a diagnostic device.

## Status
See [`CLAUDE.md`](CLAUDE.md) → "Current Implementation State" for what is built and what is not.

## Setup
```bash
python -m venv venv
pip install -e .[dev]
cp .env.example .env     # then set ANTHROPIC_API_KEY
```

## Documentation
| File | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Current state, conventions, design invariants (source of truth) |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phase-by-phase build plan |
| [`docs/architecture.md`](docs/architecture.md) | Target folder layout and import map |
| [`fedler.md`](fedler.md) | Federated learning and differential privacy plan |
| [`src/models/CLAUDE.md`](src/models/CLAUDE.md) | Model training decisions and checkpoint format |
