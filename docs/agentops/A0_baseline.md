# A0 — Budget, API baseline, and a mocked LLM

**Status: DONE.** Written 2026-11-03 from measurements already in the repo, not from
fresh runs. Every number here has a source named beside it; nothing is an estimate
unless it says so.

This is the Definition-of-Done record for Phase A0 of the Production Agent Engineering
Track. It exists because every later phase's numbers — cost per query, latency budget,
JSON-validity rate — are expressed relative to this baseline.

---

## 1. Definition of done — status

| A0 requirement | Status | Evidence |
|---|---|---|
| One real summarizer call succeeded; tokens, cost and latency recorded | **Done** 2026-10-31 | §3 below; `CLAUDE.md` known-gaps item 2 |
| `pytest` passes with network disabled, proving no test hits the API | **Done** | `pyproject.toml` `[tool.pytest.ini_options]`; §4 |
| Spend limit enforced; API key only in `.env`, which is gitignored | **Done** | `scripts/_guard.py`; `.gitignore`; §5 |
| `cost_usd()` unit-tested against hand calculations | **Done** | `src/utils/costs.py`, `tests/unit/test_costs.py` (17 tests) |

A0 was effectively completed during ordinary development rather than as a named phase,
which is why the plan document still described it as pending.

---

## 2. What was built

| Plan artefact | Actual location | Note |
|---|---|---|
| `configs/llm.yaml` price table | `configs/llm.yaml` | Verified against the pricing page 2026-10-29 |
| `src/llm/pricing.py` → `cost_usd()` | `src/utils/costs.py` | Same function, different module path |
| `tests/fakes/fake_anthropic.py` | `FakeClient` in the summarizer tests | See the caveat in §4 |
| `scripts/_guard.py` spend gate | `scripts/_guard.py` | `SpendGuard`, `SpendRefused`, `SpendLimitReached` |
| `docs/agentops/A0_baseline.md` | this file | |

**Deviation from the plan, deliberate:** the plan specifies `temperature: 0` in
`configs/llm.yaml`. That is not possible on this model. Sampling parameters are
rejected on Haiku 5.5 and are a 400 on every model from API version `2026-10-21`, so
the depth/cost control is `output_config={"effort": "low"}` instead
(`configs/agents.yaml`). Thinking is on by default, so `max_tokens` must cover
thinking **plus** the JSON body — `max_tokens: 1000` was confirmed sufficient, with
roughly 500 output tokens used in practice.

---

## 3. The measured baseline (live run, 2026-10-31)

Model: `claude-haiku-5-5`. All three `--case` values of `scripts/summarize_one.py`
were run against the real API and returned valid, parseable JSON.

| Measurement | Value |
|---|---|
| Cost per summarizer call | **~$0.0005** |
| Output tokens | ~500 (thinking + JSON body) |
| `max_tokens` needed | 1000 (sufficient, confirmed) |
| JSON parse success | 3 / 3 cases |

**Confirmed live, beyond the cost number:**

- `output_config={"effort": "low"}` is accepted.
- The disagreement caveat fires as designed.
- The artifact `recapture_needed` force fires as designed.
- On melanoma at 0.91 the model returned `urgent` **with** `immediate_referral: true`
  unprompted, so the schema validator never had to reject a response.

**Two residual unknowns, recorded honestly:**

1. The artifact case's final `confidence_level: low` is ambiguous, because `_TIER_DOWN`
   maps both `moderate→low` and `low→low`. The live run proves the branch executed
   (`recapture_needed` was forced) but not the size of the downgrade. A parametrized
   unit test covers all three tiers instead.
2. Nothing has been run against a **real lesion photo** through the summarizer. The
   ensemble and Grad-CAM have been (see `CLAUDE.md` gap 6, 15 real photos); the
   LLM leg of that path has not.

### Cost model

`cost_usd()` prices one response's `usage` from the table in `configs/llm.yaml`.
Haiku 5.5 is tiered **by prompt length**: a prompt over
`long_prompt_threshold_tokens` (100,000) pays the `long` prices on *every* category,
and **cache reads and writes count toward that length**. This is the unusual part of
the pricing model and the thing to remember when designing a cached judge prompt in
A3 — a large cached prefix can tip a request into the 5× tier.

---

## 4. Proof that no test spends money

`pyproject.toml` sets:

```toml
addopts = "--disable-socket --allow-hosts=127.0.0.1"
```

So network access is blocked for the whole suite by default; a test that genuinely
needed the network would have to declare the `enable_socket` marker, and **none do**.

`127.0.0.1` is allowed for a Windows-specific reason, not because any test talks to a
local server: `ProactorEventLoop`'s self-pipe calls `socket.socketpair()`, and with no
`AF_UNIX` available CPython falls back to a loopback `AF_INET` pair. Without that
allowance every `asyncio.run()` and every FastAPI `TestClient` call would fail.
`--allow-unix-socket` is **not** the fix, because the problem is precisely that no Unix
socket is available to fall back to.

Egress is still blocked: `connect()` to any other host raises
`SocketConnectBlockedError`, verified against a real address and through `httpx`.

**One honest limit:** the guard hooks socket creation and `connect()`, not name
resolution, so `getaddrinfo()` still succeeds. This enforces "no API calls", not
"no packets".

**Caveat on the fake client.** `FakeClient` accepts `**kwargs` and ignores them, so it
cannot catch a request parameter the real API would reject — exactly the class of bug
that `temperature` was. `test_request_carries_system_prompt_and_context` covers that
blind spot by asserting on the request *shape*: no `temperature`, `output_config`
present. The plan's richer `FakeAnthropicClient` (scripted malformed JSON, timeout,
429, empty content) is **still worth building** and belongs with the A5 failure matrix.

---

## 5. Making spending money an explicit act

`scripts/_guard.py` refuses to run any paid script unless **both**
`SKINSURE_ALLOW_SPEND=1` and an explicit `--max-usd` are set. Neither has a
money-spending default. `SpendGuard` keeps a running tally via `record()` and
`check()`, and raises `SpendLimitReached` when the cap is hit.

```bash
SKINSURE_ALLOW_SPEND=1 ./venv/Scripts/python.exe scripts/summarize_one.py \
    --case disagreement --max-usd 0.05
```

`summarize_one.py` calls `load_dotenv(override=True)`, so `.env` wins over any
`ANTHROPIC_API_KEY` already exported in the shell. `.env` is gitignored.

---

## 6. What A0 leaves open for later phases

| Item | Phase | Why it is not A0 |
|---|---|---|
| Richer `FakeAnthropicClient` (timeout / 429 / malformed / empty) | A5 | Only useful once there is a failure matrix to drive it |
| Console monthly spend limit | A0 (manual) | Set in the Anthropic Console, not in the repo — verify by hand |
| Batch API pricing path | A3 | `batch_multiplier: 0.5` is in the table but no batch call has been made |
| Prompt caching | A3 | Minimum cacheable length must be re-checked for Haiku 5.5; see the plan's caching section |
| Real lesion photo through the summarizer | A2 | Needs the replay harness |

---

## 7. Why this matters

Every A3 ship gate is a number relative to this baseline. "Cost per query ≤ $0.01" is
only meaningful because a real call was measured at ~$0.0005 — i.e. the gate has about
20× headroom, and the LLM genuinely is the dominant marginal cost. A gate written
against a guessed cost would have been either trivially passable or impossible, and
nobody would have known which.
