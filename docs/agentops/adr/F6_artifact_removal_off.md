# ADR-F6: Artifact removal stays off — measured inert

**Status:** Accepted  
**Date:** 2026-11-02  
**Decision-makers:** Manav Jain  
**Evidence:** `.nb1out/artifact_ab_analysis.md`, `.nb1out/artifact_ab_raw.csv`, `scripts/artifact_ab_experiment.py`, `scripts/artifact_ab_analyze.py`

## Context

The preprocessing pipeline includes an artifact-removal step (`src/preprocessing/artifact_removal.py`) that detects pen ink, glare/bubbles and ruler lines, then inpaints them. The v2 ensemble was trained with only resize + ImageNet normalisation — no artifact removal — so the step is a post-hoc addition whose benefit needed measurement before it could be enabled on the live path. The question: does removing artifacts before inference improve predictions enough to justify a pixel-mutating step the models never saw during training?

## Decision

Keep artifact removal **OFF** (`configs/preprocessing.yaml: steps.remove_artifacts: false`). A single discordant pair across 565 images cannot justify adding a pixel-mutating step to the inference path when the models were trained without one. The step is well-behaved — precise masks, no safety-class damage — but it is inert, and inert preprocessing is still unnecessary preprocessing.

## Evidence summary

Paired A/B on the **clean 565-image test set**, each image scored twice through the same v2 ensemble (one baseline, one with artifact removal enabled).

| Metric | Baseline (A) | Artifact removal (B) | Delta |
|---|---|---|---|
| Accuracy | 87.61% (495/565) | 87.79% (496/565) | +0.18 pp |
| Macro AUC | 0.9813 | 0.9813 | −0.0001 |
| Melanoma sensitivity | 97.58% (121/124) | 97.58% (121/124) | 0 |
| BCC sensitivity | 100% (106/106) | 100% (106/106) | 0 |

- **Fired** on 335/565 images (59.3%). Changed **exactly one label** — a fix: `v-tinea-face-4`, true Tinea, Keratosis (0.4655) → Tinea (0.4882) after inpainting 0.4% of pixels. Coin-flip margin.
- **McNemar overall:** b=0 (broke), c=1 (fixed), exact p=1.0000. Safety-class McNemar: b=0, c=0 on both melanoma and BCC.
- **Masks are precise:** fired images inpaint a median 0.31% of frame; 220/335 under 1%; median 0.03× the 10% guard cutoff. Contrast with hair removal (F5), where fired images sat at median 6.9%.
- **Guard** (`MAX_MASK_FRACTION=0.10`) stopped 24 images (4.25%, median mask 14.7%). Guard trips **never hit BCC** (0/106) — the detector is not systematically attacking a malignant class. Glare dominates (252/335 fired masks), then ink (57), ruler (26).
- **Determinism control:** 230 skipped images showed max probability shift of 0.00000000 across all classes, confirming the single flip is a genuine inpainting effect.

## Alternatives considered

1. **Enable it.** One fix, zero breaks, precise masks — the safety case is clean. Rejected because a single discordant pair on a coin-flip margin is not evidence of benefit, and adding a pixel-mutating step the models never trained on is a net risk increase for no measured gain.
2. **Enable it for specific detectors only** (e.g. glare only, since it dominates). Rejected for the same reason: the firing rate is high (59.3%) but the effect is zero on every metric that matters. Subsetting a null result does not create a positive one.

## Consequences

- `remove_artifacts` remains `false` in `configs/preprocessing.yaml`. No code change needed.
- The step is the **opposite failure mode** to hair removal (F5): F5 is mistuned and damaging; F6 is well-behaved and inert. If artifact removal is ever revisited, the case to beat is +1 image on an 0.47-margin Tinea call.
- The `MAX_MASK_FRACTION` guard is doing real work — without it, 63.5% of images would be inpainted — so the guard must not be weakened if the step is later enabled.
- **What would reopen this:** models retrained with artifact removal in the augmentation pipeline (so train and inference match), or a larger test set where the effect size could be resolved.
