# ADR-F4: Ranked-melanoma escalation rule (prototype, off by default)

**Status:** Accepted  
**Date:** 2026-11-02  
**Decision-makers:** Manav Jain  
**Evidence:** `.nb1out/f4_threshold_sweep_val.csv`, `.nb1out/weighting_melanoma_misses.csv`, `src/agents/severity_triage.py`

## Context

F1's investigation of ensemble weights revealed that all three melanomas the shipped ECS ensemble misses share one shape: melanoma ranks high but is **not** the argmax (p(melanoma) = 0.343, 0.369, 0.064). The existing Rule 2 in `severity_triage.py` gates on `primary_label` — the argmax — so a melanoma ranked second or lower can never trigger it, regardless of threshold. This is a structural blind spot: the rule cannot see the cases it most needs to catch.

The question was whether a new rule escalating on `p(melanoma)` regardless of rank could close the gap safely, and if so, whether the evidence justified turning it on.

## Decision

Implement Rule F4 as a **config-gated prototype, OFF by default** (`triage.escalate_on_ranked_melanoma: true` in `configs/agents.yaml`). The code ships; the behaviour does not. The threshold is 0.31.

The rule reads `Predictions.class_probabilities` via `probability_of()`. When that field is absent (e.g. in tests with minimal fakes), the rule falls back to the top-2 fields and simply does not fire for a melanoma ranked third or lower — preserving today's behaviour exactly.

## Evidence summary

**Threshold selection (748-image validation split, 202 melanoma, 546 non-melanoma):**

- Sweep from 0.10 to 0.45 in steps of 0.01 (`.nb1out/f4_threshold_sweep_val.csv`).
- At **t = 0.31**: recovers all 3 non-argmax validation melanomas at **zero** false escalations on 546 non-melanoma images.
- Zero-cost plateau spans 0.31–0.35. At 0.36, recall drops from 3 to 1.
- At t = 0.30: same recall but 1 false escalation on validation — validation independently confirms 0.31 as the clean cut.

**Test-set verification (565-image clean test set, 124 melanoma, 441 non-melanoma):**

- Recovers **2 of 3** known misses (p = 0.343, 0.369) for **1** false escalation (0.23% of non-melanoma).
- The third miss (p = 0.064) is unreachable at any usable threshold.
- Note: t = 0.30 was the value originally fitted to these 3 test cases. Validation independently agreed the neighbourhood is right but placed the clean boundary one step higher.

**Why OFF:** 3 validation melanomas and 3 test melanomas cannot establish that this is a net safety gain. The effect size is real on paper — recovering 2 missed melanomas for 1 false flag — but the sample is too small to distinguish signal from noise, and a false escalation in a screening tool has its own cost (unnecessary referrals, patient anxiety). Shipping it on would mean acting on an unmeasured benefit.

## Alternatives considered

1. **Change ensemble weights to ES (drop ConvNeXt).** Rejected in F1: the melanoma gain is not significant (McNemar p = 0.50), while the accuracy cost is (p = 0.043), and BCC sensitivity regresses on validation (−3.36 pp). F4 recovers the same melanoma cases without touching weights.

2. **Lower Rule 2's threshold below 0.40.** Irrelevant — Rule 2 gates on `primary_label`, so no threshold change can reach a melanoma that isn't the argmax. The structural gap is in the gating condition, not the threshold.

3. **Ship F4 ON by default.** Premature given the sample size. The rule only escalates (never downgrades or blocks), so the worst case if turned on is unnecessary referrals, not missed melanomas — but "low risk of harm" is not the same as "evidence of benefit."

## Consequences

- **Residual risk while OFF:** the 2 recoverable melanomas (p ≈ 0.34–0.37) continue to be missed by triage. This is the same risk as before F4 existed — the rule adds an option, it doesn't remove a safety net.
- **What would turn it ON:** a larger melanoma-heavy test set (≥500 melanoma images) showing the rule recovers missed cases without a disproportionate false-escalation rate; or a clinical decision that any melanoma at p ≥ 0.31 warrants referral regardless of statistical power.
- **What would change the threshold:** a different dataset with different class-probability distributions. The 0.31–0.35 plateau is specific to these v2 checkpoints; retraining the ensemble invalidates the sweep.
- **Implementation detail:** the rule is skipped when the result is already blocked, logs a distinct `reasons` entry, and follows the "only escalate or block, never downgrade" invariant.
