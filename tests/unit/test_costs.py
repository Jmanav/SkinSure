"""Cost accounting: the price table, the prompt-length tiers, and reading `response.usage`."""
import asyncio

import pytest
from helpers import FakeClient, llm_json

from src.agents.clinical_summarizer import summarize
from src.utils.costs import cost_usd

# A fixture table, so these tests pin the arithmetic rather than today's real prices.
# Round numbers make every expected value checkable by hand.
PRICES = {
    "llm": {"model": "test-model"},
    "prices_usd_per_mtok": {
        "test-model": {
            "long_prompt_threshold_tokens": 1000,
            "batch_multiplier": 0.5,
            "short": {"input": 1.0, "output": 10.0, "cache_read": 0.1, "cache_write_5m": 1.25, "cache_write_1h": 2.0},
            "long": {"input": 5.0, "output": 50.0, "cache_read": 0.5, "cache_write_5m": 6.25, "cache_write_1h": 10.0},
        }
    },
}


def _cost(**kwargs):
    return cost_usd(config=PRICES, **kwargs)


# --- pricing arithmetic ------------------------------------------------------------------------

def test_short_prompt_uses_short_prices():
    # 100 in @ $1/Mtok + 10 out @ $10/Mtok = (100 + 100) / 1e6
    assert _cost(input_tokens=100, output_tokens=10) == pytest.approx(200 / 1e6)


def test_zero_tokens_cost_nothing():
    assert _cost(input_tokens=0, output_tokens=0) == 0.0


def test_batch_halves_the_cost():
    plain = _cost(input_tokens=100, output_tokens=10)
    assert _cost(input_tokens=100, output_tokens=10, batch=True) == pytest.approx(plain * 0.5)


def test_cache_write_is_priced_by_ttl():
    five_min = _cost(input_tokens=0, output_tokens=0, cache_write=100)
    one_hour = _cost(input_tokens=0, output_tokens=0, cache_write=100, cache_ttl="1h")
    assert five_min == pytest.approx(125 / 1e6)
    assert one_hour == pytest.approx(200 / 1e6)
    assert one_hour > five_min


def test_cache_read_is_cheaper_than_fresh_input():
    assert _cost(input_tokens=0, output_tokens=0, cache_read=100) < _cost(
        input_tokens=100, output_tokens=0
    )


# --- the prompt-length tier (the part that is easy to get wrong) -------------------------------

def test_threshold_is_inclusive_so_a_prompt_exactly_at_it_stays_short():
    assert _cost(input_tokens=1000, output_tokens=0) == pytest.approx(1000 / 1e6)


def test_one_token_over_the_threshold_reprices_the_whole_request():
    # Not just the excess: all 1001 input tokens jump to the long rate.
    assert _cost(input_tokens=1001, output_tokens=0) == pytest.approx(1001 * 5 / 1e6)


def test_output_tokens_do_not_count_toward_prompt_length():
    # 500 in + 5000 out stays short: only input decides the tier.
    assert _cost(input_tokens=500, output_tokens=5000) == pytest.approx((500 + 50_000) / 1e6)


def test_cached_tokens_count_toward_prompt_length():
    # 600 fresh + 600 cache-read = 1200 > 1000, so this is a long prompt even though
    # neither figure exceeds the threshold alone. This is the rule most likely to be
    # missed, and it makes the request 5x more expensive per token.
    cost = _cost(input_tokens=600, output_tokens=0, cache_read=600)
    assert cost == pytest.approx((600 * 5 + 600 * 0.5) / 1e6)


def test_cache_writes_count_toward_prompt_length_too():
    cost = _cost(input_tokens=600, output_tokens=0, cache_write=600)
    assert cost == pytest.approx((600 * 5 + 600 * 6.25) / 1e6)


# --- guards ------------------------------------------------------------------------------------

def test_negative_tokens_are_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        _cost(input_tokens=-1, output_tokens=0)


def test_unknown_cache_ttl_is_rejected():
    with pytest.raises(ValueError, match="cache_ttl"):
        _cost(input_tokens=1, output_tokens=1, cache_ttl="7d")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="price table"):
        _cost(input_tokens=1, output_tokens=1, model="no-such-model")


def test_real_config_prices_the_configured_model():
    # No `config=`, so this reads configs/llm.yaml and proves the shipped table has
    # every key cost_usd() needs for the model the app is actually pointed at.
    assert cost_usd(input_tokens=1000, output_tokens=1000) > 0


# --- usage off a response ----------------------------------------------------------------------

def _summarize_with(client, make_predictions, make_patient, make_xai):
    """Drive one real summarize() call so `usage` comes off a genuine response path."""
    return asyncio.run(summarize(make_predictions(), make_patient(), make_xai(), client=client))


def test_fake_client_reports_usage_that_cost_usd_can_price(
    make_predictions, make_patient, make_xai
):
    """The fake's usage block is shaped like the SDK's, so it can be priced directly."""
    client = FakeClient(llm_json(), usage={"input_tokens": 800, "output_tokens": 200})
    _summarize_with(client, make_predictions, make_patient, make_xai)

    usage = client.last_usage
    assert _cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read=usage.cache_read_input_tokens,
        cache_write=usage.cache_creation_input_tokens,
    ) == pytest.approx((800 + 2000) / 1e6)


def test_cache_fields_default_to_zero_not_missing(make_predictions, make_patient, make_xai):
    # A caller reading cache_read_input_tokens off an uncached real response gets 0,
    # so the fake must too -- otherwise cost_usd() would see an AttributeError here
    # and pass against a response shape the API never returns.
    client = FakeClient(llm_json())
    _summarize_with(client, make_predictions, make_patient, make_xai)

    usage = client.last_usage
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_usage_field_names_match_the_real_sdk():
    """Guards the fake against SDK drift: a rename here breaks cost accounting silently."""
    from anthropic.types import Usage

    client = FakeClient(llm_json())
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens"):
        assert field in Usage.model_fields, f"{field} is not an SDK Usage field"
        assert hasattr(client.last_usage_template, field)
