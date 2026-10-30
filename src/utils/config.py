from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from src.models.ensemble import CheckpointPaths

_ROOT = Path(__file__).parent.parent.parent
_AGENTS_YAML = _ROOT / "configs" / "agents.yaml"
_MODELS_YAML = _ROOT / "configs" / "models.yaml"
_PREPROCESSING_YAML = _ROOT / "configs" / "preprocessing.yaml"


class SummarizerConfig(BaseModel):
    model: str
    max_tokens: int
    effort: str = "low"


class AgentsConfig(BaseModel):
    summarizer: SummarizerConfig


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


def get_checkpoint_paths() -> CheckpointPaths:
    """Checkpoint paths from configs/models.yaml, resolved against the project root."""
    from src.models.ensemble import CheckpointPaths  # deferred: importing it pulls in torch

    ckpt = get_model_config().checkpoints
    return CheckpointPaths(
        efficientnet=_ROOT / ckpt["efficientnet"],
        convnext=_ROOT / ckpt["convnext"],
        swin=_ROOT / ckpt["swin"],
    )
