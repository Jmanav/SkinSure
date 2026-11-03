from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from src.models.ensemble import CheckpointPaths

_ROOT = Path(__file__).parent.parent.parent
_AGENTS_YAML = _ROOT / "configs" / "agents.yaml"
_MODELS_YAML = _ROOT / "configs" / "models.yaml"
_PREPROCESSING_YAML = _ROOT / "configs" / "preprocessing.yaml"
_OBSERVABILITY_YAML = _ROOT / "configs" / "observability.yaml"


class SummarizerConfig(BaseModel):
    model: str
    max_tokens: int
    effort: str = "low"


class TriageConfig(BaseModel):
    # Rule F4: escalate when melanoma is ranked below top-1 but still probable.
    # Default OFF - it changes clinical escalation behaviour. Threshold selected on
    # the validation split; see configs/agents.yaml for the criterion.
    escalate_on_ranked_melanoma: bool = False
    ranked_melanoma_min_confidence: float = 0.31


class AgentsConfig(BaseModel):
    summarizer: SummarizerConfig
    triage: TriageConfig = TriageConfig()


class Settings(BaseSettings):
    agents: AgentsConfig
    model_config = {"arbitrary_types_allowed": True}


class ModelConfig(BaseModel):
    checkpoints: dict[str, str]
    weights: dict[str, float]


class QualityConfig(BaseModel):
    min_short_side: int
    blur_reject_below: float
    blur_warn_below: float
    blur_measure_max_side: int


class StepsConfig(BaseModel):
    remove_hair: bool = False
    remove_artifacts: bool = False
    normalize_color: bool = False


class PreprocessingConfig(BaseModel):
    quality: QualityConfig
    steps: StepsConfig


@lru_cache(maxsize=1)
def get_preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(**yaml.safe_load(_PREPROCESSING_YAML.read_text()))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = yaml.safe_load(_AGENTS_YAML.read_text())
    return Settings(agents=AgentsConfig(**raw))


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    return ModelConfig(**yaml.safe_load(_MODELS_YAML.read_text()))


class ObservabilityConfig(BaseModel):
    # Default OFF: tracing is opt-in, the observability extra may not be installed, and
    # a clinical API must boot without a telemetry stack. See configs/observability.yaml
    # for the full reasoning. Literal rather than str so an unknown exporter fails at
    # load time with a clear error, not at configure_tracing() on the request path.
    exporter: Literal["none", "console", "otlp"] = "none"
    service_name: str = "skinsure"


@lru_cache(maxsize=1)
def get_observability_config() -> ObservabilityConfig:
    return ObservabilityConfig(**yaml.safe_load(_OBSERVABILITY_YAML.read_text()))


def get_checkpoint_paths() -> CheckpointPaths:
    """Checkpoint paths from configs/models.yaml, resolved against the project root."""
    from src.models.ensemble import CheckpointPaths  # deferred: importing it pulls in torch

    ckpt = get_model_config().checkpoints
    return CheckpointPaths(
        efficientnet=_ROOT / ckpt["efficientnet"],
        convnext=_ROOT / ckpt["convnext"],
        swin=_ROOT / ckpt["swin"],
    )
