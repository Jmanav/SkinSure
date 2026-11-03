# ADR-F1: Keep equal ensemble weights

**Status:** Accepted
**Date:** 2026-11-02
**Decision-makers:** Manav Jain
**Evidence:** `.nb1out/ensemble_weighting_analysis.md`, `.nb1out/weighting_*.csv`, `.nb1out/per_class_metrics.csv`

## Context

The shipped ensemble averages EfficientNet-B1, ConvNeXt-Tiny and Swin-Tiny with equal weights (1/1/1). A grid search over 1,033 weight vectors on the clean 565-image test set showed that the EfficientNet + Swin (ES) pair reaches 99.19% melanoma sensitivity versus ECS's 97.58% — a 1.6 pp gap on the single most important metric for a melanoma screening tool. The question was whether to drop ConvNeXt to capture that gain.

## Decision

Keep the equal-weight three-backbone (ECS) configuration. The melanoma sensitivity difference is not statistically distinguishable at this sample size, while the cost of switching is measured and significant. A cheaper, no-trade-off lever — a triage rule that escalates melanoma ranked below top-1 — recovers the same sensitivity without changing the ensemble (see ADR-F4).

## Evidence summary

All figures on the de-duplicated **565-image clean test set** unless stated otherwise.

**The melanoma gap is noise at this n.** ECS misses 3 melanomas (121/124 = 97.58%), ES misses 1 (123/124 = 99.19%). Only 2 images are discordant (ECS wrong, ES right; zero the other way). McNemar exact p = 0.50. Wilson 95% CIs overlap: ECS [93.13%, 99.17%], ES [95.57%, 99.86%]. Pooling test + validation (326 melanoma images) gives 5 unanimously ES-favouring discordances — a perfectly one-sided result, but the best achievable p is 0.0625 (need ≥ 6 for α = 0.05). Reaching 80% power at a 1.5 pp effect requires ~1,500–2,000 melanoma images; we have 124 (or 326 pooled).

**Better-powered statistics favour ECS.** Macro AUC (all 565 images, threshold-free): ECS 0.9813 vs ES 0.9772. Melanoma-vs-rest AUC: ECS 0.9987 vs ES 0.9966. ECS→ES accuracy drop: −1.95 pp on test (McNemar b = 18, c = 7, exact p = 0.043), −2.01 pp on validation (b = 26, c = 11, p = 0.020). ES is worse on 4 of 5 validation classes including BCC (−3.36 pp), the secondary safety target.

## Alternatives considered

| Alternative | Outcome |
|---|---|
| Switch to ES (drop ConvNeXt) | Measured 2 pp accuracy cost (significant), BCC regression on validation, for an unmeasured melanoma benefit. Rejected: paying a measured cost for an unmeasured benefit is the wrong trade. |
| Tune weights on validation, report on test | 28.3% of weight vectors hit the same best melanoma count, so the choice is mostly noise. Grid-search ranking on test is optimistically biased by construction. Rejected as unreliable. |
| Triage rule for ranked melanoma (p(mel) ≥ threshold) | Recovers 2 of 3 missed melanomas at 1 false escalation (0.23% of non-melanoma), no accuracy loss, no weight change. Adopted as ADR-F4 instead. |

## Consequences

- `configs/models.yaml` stays at `efficientnet: 1, convnext: 1, swin: 1`.
- **Residual risk:** the ~1.5 pp melanoma effect may be real — ECS may miss roughly 1 melanoma in 65 that ES catches. This is partially mitigated by Rule F4 (ADR-F4), which recovers 2 of the 3 specific misses through triage rather than ensemble changes.
- **Reopener:** a dataset with ≥ 1,500 melanoma images, or a new backbone/checkpoint generation, would justify re-evaluation. Tune on validation, report on test.
- **Paper value:** publishable as a null result with a power analysis — a corrective to ablation tables that rank configurations on fractions of a percent without testing significance.
