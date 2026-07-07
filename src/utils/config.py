from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).parent.parent.parent
_AGENTS_YAML = _REPO_ROOT / "configs" / "agents.yaml"
_MODELS_YAML = _REPO_ROOT / "configs" / "models.yaml"


class SummarizerConfig(BaseModel):
    model: str
    max_tokens: int
    temperature: float


class AgentsConfig(BaseModel):
    summarizer: SummarizerConfig


class Settings(BaseSettings):
    agents: AgentsConfig

    model_config = {"arbitrary_types_allowed": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = yaml.safe_load(_AGENTS_YAML.read_text())
    return Settings(agents=AgentsConfig(**raw))


class ModelPathsConfig(BaseModel):
    checkpoints: dict[str, str]
    weights: dict[str, float]


@lru_cache(maxsize=1)
def get_model_config() -> ModelPathsConfig:
    raw = yaml.safe_load(_MODELS_YAML.read_text())
    return ModelPathsConfig(**raw)


def get_checkpoint_paths() -> "CheckpointPaths":
    from src.models.ensemble import CheckpointPaths

    cfg = get_model_config().checkpoints
    return CheckpointPaths(
        efficientnet=_REPO_ROOT / cfg["efficientnet"],
        convnext=_REPO_ROOT / cfg["convnext"],
        swin=_REPO_ROOT / cfg["swin"],
    )
