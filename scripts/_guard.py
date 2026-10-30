"""Spend gate for scripts that call a paid API.

Two independent locks, both of which must be opened deliberately:

    SKINSURE_ALLOW_SPEND=1      in the environment
    --max-usd <cap>             on the command line

Neither has a default that spends money. The env var cannot be set by accident in
a script's own code, and the cap cannot be supplied by a default, so a script that
forgets to pass one refuses to run rather than running uncapped.

Usage::

    from scripts._guard import SpendGuard, add_spend_args

    parser = argparse.ArgumentParser()
    add_spend_args(parser)
    args = parser.parse_args()

    guard = SpendGuard.from_args(args)       # raises SpendRefused if not authorised
    ...
    guard.record(response.usage)             # raises SpendLimitReached at the cap

`record()` is called *after* a request, because the cost is only known from the
response. The cap therefore bounds what the next request is allowed to start, not
what the last one was allowed to finish: a single request can carry the tally past
the cap, and the run stops before another begins. Sizing the cap to leave room for
one request over is the caller's job.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from src.utils.costs import cost_usd

ALLOW_SPEND_ENV = "SKINSURE_ALLOW_SPEND"


class SpendRefused(RuntimeError):
    """The run was never authorised: the env var or the cap is missing or invalid."""


class SpendLimitReached(RuntimeError):
    """The tally reached the cap. Raised by `record()`, after the request it prices."""


def add_spend_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--max-usd` to a parser.

    Deliberately has no default: argparse leaves it None, and `from_args` refuses.
    A default here -- even a small one -- would mean a script could spend money
    without anyone naming an amount.
    """
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help="Hard spend cap for this run, in USD. Required; there is no default.",
    )
    return parser


@dataclass
class SpendGuard:
    """Tracks spend against a cap. Build with `from_args`, not directly."""

    max_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    _costs: list[float] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, env: dict | None = None) -> SpendGuard:
        """Check both locks and return a guard, or raise `SpendRefused`.

        `env` is injectable so tests need not mutate os.environ.
        """
        environ = os.environ if env is None else env

        if environ.get(ALLOW_SPEND_ENV) != "1":
            raise SpendRefused(
                f"This script spends real money and is not authorised to run.\n"
                f"Set {ALLOW_SPEND_ENV}=1 to allow it, and pass --max-usd <cap>.\n"
                f"(found {ALLOW_SPEND_ENV}={environ.get(ALLOW_SPEND_ENV)!r})"
            )

        max_usd = getattr(args, "max_usd", None)
        if max_usd is None:
            raise SpendRefused(
                "No spend cap given. Pass --max-usd <cap> to set the maximum this run "
                "may spend. There is deliberately no default."
            )
        if max_usd <= 0:
            raise SpendRefused(f"--max-usd must be greater than 0, got {max_usd}.")

        return cls(max_usd=float(max_usd))

    @property
    def remaining_usd(self) -> float:
        """Headroom left under the cap; never negative."""
        return max(0.0, self.max_usd - self.spent_usd)

    def record(self, usage, **kwargs) -> float:
        """Price one response's `usage`, add it to the tally, and return its cost.

        Raises `SpendLimitReached` once the tally is at or above the cap. Extra kwargs
        (`model`, `batch`, `cache_ttl`, `config`) pass through to `cost_usd`.

        Reads the cache fields with getattr defaults so a response that omits them --
        any uncached call -- prices as 0 rather than raising.
        """
        cost = cost_usd(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            **kwargs,
        )
        self.spent_usd += cost
        self.calls += 1
        self._costs.append(cost)

        if self.spent_usd >= self.max_usd:
            raise SpendLimitReached(
                f"Spend cap reached: ${self.spent_usd:.6f} of ${self.max_usd:.6f} "
                f"after {self.calls} call(s). Stopping before the next request."
            )
        return cost

    def check(self) -> None:
        """Raise if the cap is already reached, without pricing anything.

        For a loop that wants to stop before paying for a request it cannot afford.
        """
        if self.spent_usd >= self.max_usd:
            raise SpendLimitReached(
                f"Spend cap reached: ${self.spent_usd:.6f} of ${self.max_usd:.6f} "
                f"after {self.calls} call(s)."
            )

    def summary(self) -> str:
        """One line for the end of a run, whether it finished or was cut short."""
        return (
            f"{self.calls} call(s), ${self.spent_usd:.6f} of ${self.max_usd:.6f} cap "
            f"(${self.remaining_usd:.6f} left)"
        )
