from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

_AGENTS_YAML = Path(__file__).parent.parent.parent / "configs" / "agents.yaml"


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
