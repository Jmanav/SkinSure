"""The spend gate: both locks, the tally, and the abort at the cap."""
import argparse
from types import SimpleNamespace

import pytest

from scripts._guard import (
    ALLOW_SPEND_ENV,
    SpendGuard,
    SpendLimitReached,
    SpendRefused,
    add_spend_args,
)

ALLOWED = {ALLOW_SPEND_ENV: "1"}

# Priced so the arithmetic is obvious: $1/Mtok in, $10/Mtok out, no tier change
# within these tests. 1000 in + 100 out = $0.002 per call.
PRICES = {
    "llm": {"model": "test-model"},
    "prices_usd_per_mtok": {
        "test-model": {
            "long_prompt_threshold_tokens": 10_000_000,
            "batch_multiplier": 0.5,
            "short": {"input": 1.0, "output": 10.0, "cache_read": 0.1,
                      "cache_write_5m": 1.25, "cache_write_1h": 2.0},
            "long": {"input": 5.0, "output": 50.0, "cache_read": 0.5,
                     "cache_write_5m": 6.25, "cache_write_1h": 10.0},
        }
    },
}

COST_PER_CALL = (1000 * 1.0 + 100 * 10.0) / 1e6  # $0.002


def usage(input_tokens=1000, output_tokens=100, **extra):
    """A response.usage, shaped like the SDK's."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=extra.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=extra.get("cache_creation_input_tokens", 0),
    )


def args(max_usd=1.0):
    return argparse.Namespace(max_usd=max_usd)


def guard(max_usd=1.0, env=ALLOWED):
    return SpendGuard.from_args(args(max_usd), env=env)


def record(g, **kw):
    return g.record(usage(**kw), config=PRICES)


# --- lock 1: the environment variable ----------------------------------------------------------

def test_refuses_when_env_var_is_absent():
    with pytest.raises(SpendRefused, match=ALLOW_SPEND_ENV):
        SpendGuard.from_args(args(), env={})


@pytest.mark.parametrize("value", ["0", "", "true", "TRUE", "yes", "2", " 1", "1 "])
def test_only_exactly_1_authorises(value):
    # Anything truthy-looking must not pass: "true" and "yes" are the values someone
    # reaches for from memory, and a near-miss must fail closed, not open.
    with pytest.raises(SpendRefused):
        SpendGuard.from_args(args(), env={ALLOW_SPEND_ENV: value})


def test_accepts_exactly_1():
    assert SpendGuard.from_args(args(2.5), env=ALLOWED).max_usd == 2.5


# --- lock 2: the cap ---------------------------------------------------------------------------

def test_refuses_when_cap_is_missing():
    with pytest.raises(SpendRefused, match="No spend cap"):
        SpendGuard.from_args(args(max_usd=None), env=ALLOWED)


def test_refuses_when_cap_is_missing_even_though_env_var_is_set():
    # The locks are independent: authorising spend does not imply a cap.
    with pytest.raises(SpendRefused):
        SpendGuard.from_args(argparse.Namespace(max_usd=None), env=ALLOWED)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.01])
def test_refuses_a_non_positive_cap(bad):
    # A cap of 0 would abort on the first record() anyway, but refusing up front
    # says why instead of looking like a mid-run failure.
    with pytest.raises(SpendRefused, match="greater than 0"):
        SpendGuard.from_args(args(bad), env=ALLOWED)


def test_refuses_when_the_arg_was_never_added_to_the_parser():
    # A script that forgets add_spend_args() must refuse, not run uncapped.
    with pytest.raises(SpendRefused, match="No spend cap"):
        SpendGuard.from_args(argparse.Namespace(), env=ALLOWED)


# --- the parser contract -----------------------------------------------------------------------

def test_max_usd_has_no_default():
    # The crucial property: argparse must leave this None so from_args refuses.
    # A default here would let a script spend without anyone naming an amount.
    parsed = add_spend_args(argparse.ArgumentParser()).parse_args([])
    assert parsed.max_usd is None


def test_max_usd_parses_as_float():
    parsed = add_spend_args(argparse.ArgumentParser()).parse_args(["--max-usd", "0.25"])
    assert parsed.max_usd == 0.25


def test_parser_and_guard_compose():
    parsed = add_spend_args(argparse.ArgumentParser()).parse_args(["--max-usd", "5"])
    assert SpendGuard.from_args(parsed, env=ALLOWED).max_usd == 5.0


# --- the tally ---------------------------------------------------------------------------------

def test_a_fresh_guard_has_spent_nothing():
    g = guard()
    assert (g.spent_usd, g.calls) == (0.0, 0)


def test_record_returns_the_cost_of_that_call():
    assert record(guard()) == pytest.approx(COST_PER_CALL)


def test_tally_accumulates_across_calls():
    g = guard()
    for _ in range(3):
        record(g)
    assert g.spent_usd == pytest.approx(3 * COST_PER_CALL)
    assert g.calls == 3


def test_cached_tokens_are_included_in_the_tally():
    g = guard()
    cost = record(g, cache_read_input_tokens=1000)
    assert cost == pytest.approx(COST_PER_CALL + 1000 * 0.1 / 1e6)


def test_usage_without_cache_fields_prices_as_zero():
    # A response that omits the cache fields must price, not raise: getattr defaults
    # exist so an uncached call from any SDK version still works.
    g = guard()
    bare = SimpleNamespace(input_tokens=1000, output_tokens=100)
    assert g.record(bare, config=PRICES) == pytest.approx(COST_PER_CALL)


def test_remaining_tracks_the_tally():
    g = guard(max_usd=1.0)
    record(g)
    assert g.remaining_usd == pytest.approx(1.0 - COST_PER_CALL)


# --- the abort ---------------------------------------------------------------------------------

def test_aborts_once_the_tally_reaches_the_cap():
    # Cap at 2.5 calls' worth: calls 1 and 2 pass, call 3 crosses and raises.
    g = guard(max_usd=COST_PER_CALL * 2.5)
    record(g)
    record(g)
    with pytest.raises(SpendLimitReached):
        record(g)


def test_abort_is_at_or_above_not_strictly_above():
    # Cap exactly equal to one call: that call must be the last. "Reaching" the cap
    # stops the run; it does not have to be exceeded.
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached):
        record(g)


def test_the_call_that_crosses_is_still_counted():
    # record() prices a request that already happened, so its cost is real money and
    # must stay in the tally even though it triggered the abort.
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached):
        record(g)
    assert g.spent_usd == pytest.approx(COST_PER_CALL)
    assert g.calls == 1


def test_the_error_names_the_tally_and_the_cap():
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached) as exc:
        record(g)
    assert "0.002000" in str(exc.value)


def test_a_single_expensive_call_can_overshoot_the_cap():
    # Documented limitation: the cost is only known after the request, so one call
    # can carry the tally past the cap. The guard bounds the NEXT request, not the
    # one that already ran.
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached):
        g.record(usage(input_tokens=1_000_000, output_tokens=0), config=PRICES)
    assert g.spent_usd > g.max_usd
    assert g.remaining_usd == 0.0  # clamped, never negative


def test_recording_after_the_abort_raises_again():
    # A caller that catches the abort and keeps going must not get a free call.
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached):
        record(g)
    with pytest.raises(SpendLimitReached):
        record(g)


# --- check(): stop before paying ---------------------------------------------------------------

def test_check_passes_while_under_the_cap():
    guard(max_usd=1.0).check()  # does not raise


def test_check_raises_at_the_cap_without_pricing_anything():
    g = guard(max_usd=COST_PER_CALL)
    with pytest.raises(SpendLimitReached):
        record(g)
    before = g.calls
    with pytest.raises(SpendLimitReached):
        g.check()
    assert g.calls == before  # check() is not a call


# --- the summary line --------------------------------------------------------------------------

def test_summary_reports_calls_spend_and_cap():
    g = guard(max_usd=1.0)
    record(g)
    line = g.summary()
    assert "1 call(s)" in line
    assert "0.002000" in line
    assert "1.000000" in line


# --- the real price table ----------------------------------------------------------------------

def test_prices_against_the_shipped_config_when_none_is_injected():
    # No config=, so this reads configs/llm.yaml and proves the guard works against
    # the table the app actually ships with.
    g = guard(max_usd=100.0)
    assert g.record(usage()) > 0


# --- the wiring in summarize_one.py ------------------------------------------------------------

def test_summarize_one_refuses_without_authorisation(monkeypatch, capsys):
    """The guard must gate the real-API script before any client is constructed."""
    import asyncio

    from scripts import summarize_one

    monkeypatch.delenv(ALLOW_SPEND_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["summarize_one.py", "--max-usd", "1.0"])

    with pytest.raises(SpendRefused):
        asyncio.run(summarize_one.main())


def test_summarize_one_refuses_without_a_cap(monkeypatch):
    import asyncio

    from scripts import summarize_one

    monkeypatch.setenv(ALLOW_SPEND_ENV, "1")
    monkeypatch.setattr("sys.argv", ["summarize_one.py"])

    with pytest.raises(SpendRefused, match="No spend cap"):
        asyncio.run(summarize_one.main())


def test_metered_client_captures_usage():
    """_MeteredClient must pass the response through and keep its usage."""
    import asyncio

    from scripts.summarize_one import _MeteredClient

    expected = usage()
    inner = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kw: _done(SimpleNamespace(content=[], usage=expected))
        )
    )
    client = _MeteredClient(inner)
    returned = asyncio.run(client.messages.create(model="m"))

    assert returned.usage is expected
    assert client.usage is expected


async def _done(value):
    return value
