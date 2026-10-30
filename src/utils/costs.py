"""USD cost accounting for Claude API calls, priced from configs/llm.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "llm.yaml"

_CACHE_TTLS = ("5m", "1h")


@lru_cache
def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
    batch: bool = False,
    *,
    model: str | None = None,
    cache_ttl: str = "5m",
    config: dict | None = None,
) -> float:
    """Return the USD cost of one request.

    Token counts come straight off `response.usage`. `config` is injectable so tests
    can price against a fixture instead of the real table.
    """
    if min(input_tokens, output_tokens, cache_read, cache_write) < 0:
        raise ValueError("token counts must be >= 0")

    if cache_ttl not in _CACHE_TTLS:
        raise ValueError(f"cache_ttl must be one of {_CACHE_TTLS}, got {cache_ttl!r}")

    cfg = config or _load_config()
    model = model or cfg["llm"]["model"]
    try:
        block = cfg["prices_usd_per_mtok"][model]
    except KeyError:
        raise ValueError(f"No price table for model {model!r}") from None

    # Prompt length counts every input token, cache reads and writes included.
    prompt_tokens = input_tokens + cache_read + cache_write
    tier = "short" if prompt_tokens <= block["long_prompt_threshold_tokens"] else "long"
    p = block[tier]

    total = (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_read * p["cache_read"]
        + cache_write * p[f"cache_write_{cache_ttl}"]
    ) / 1_000_000

    if batch:
        total *= block["batch_multiplier"]
    return total
