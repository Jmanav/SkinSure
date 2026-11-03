# ADR-F5: Hair removal stays off — measured harmful

**Status:** Accepted  
**Date:** 2026-11-02  
**Decision-makers:** Manav Jain  
**Evidence:** `.nb1out/hair_ab_raw.csv`, `scripts/hair_ab_experiment.py`, `scripts/hair_ab_analyze.py`

## Context

The preprocessing pipeline includes a DullRazor-style hair removal step (black-hat morphological filter + inpainting, `src/preprocessing/hair_removal.py`). It was inherited from a standard dermatology pipeline and shipped OFF by default because the v2 ensemble was trained without it. The question was whether enabling it would improve predictions on the 565-image clean test set.

## Decision

Hair removal stays **off**. It is not merely unhelpful — it is actively harmful, preferentially damaging the BCC class (a malignant target). The step is **mistuned, not just ineffective**: the black-hat filter at `threshold=10` cannot distinguish hair from dark, textured lesion tissue, so it fires disproportionately on BCC and inpaints over the lesion itself.

## Evidence summary

Paired A/B over all 565 clean test images (de-duplicated, primary reporting basis), each image scored twice through the same v2 ensemble.

**The step never corrected a single image and broke two**, both true BCC:

| Image | True class | A (baseline) | B (hair removed) | Effect |
|---|---|---|---|---|
| `ISIC_0054063` | BCC | BCC at 0.9314 | Keratosis at 0.3366 | Confident malignant → benign |
| `ISIC_0073116` | BCC | BCC | Melanoma | Wrong malignant class |

Whole-set impact: accuracy −0.35 pp, macro AUC −0.0011, **BCC sensitivity 100% → 98.11%**. On the 41 images where it fired: accuracy −4.88 pp, BCC sensitivity −7.69 pp. Melanoma sensitivity was **exactly unchanged** (121/124 both arms, including all 7 melanomas where it fired). McNemar: b=2, c=0, exact p=0.50 — not significant at n=2, but the sign is unanimous (nothing was ever fixed).

**Root cause:** at `threshold=10` the black-hat filter is non-selective. Of 524 images where `MAX_MASK_FRACTION` blocked inpainting, the mask covered a median 58.9% of the frame (≥75% on 185 of them) — the guard is the only thing preventing wholesale pixel destruction. The 41 images (7.26%) that slip under the guard inpainted a median 6.9% of pixels, sitting just below the 10% cutoff rather than being precise hair masks. Firing concentrates on BCC (24.5% firing rate) because BCC lesions are dark and textured and respond to black-hat like hair does.

**Determinism control:** 524 skipped images showed max probability shift of 0.00000 across all classes, confirming the scoring path is deterministic and both flips are genuine inpainting effects.

## Alternatives considered

1. **Enable it anyway** — rejected: unanimous harm, zero benefit, and the damage falls on a malignant class.
2. **Re-tune the threshold higher** — not pursued: a threshold change alone cannot fix a kernel that responds to lesion texture the same way it responds to hair. Would need a thinner, elongation-aware kernel validated against hair-annotated images.
3. **Replace with a learned hair segmenter** — out of scope: no hair annotations exist in this dataset, and the models were trained without the step.

## Consequences

- `configs/preprocessing.yaml` keeps `remove_hair: false`. The code stays in-tree as a reference implementation.
- Any future attempt to re-enable must first re-tune the kernel (higher threshold, elongation-aware morphology) and re-run the paired A/B on the full test set, checking BCC sensitivity specifically.
- The contrast with artifact removal (F6) is instructive: that step fires 8× more often yet is inert, because its masks are precise (median 0.31% of frame vs hair's 6.9%). Mask precision, not firing rate, is what separates a safe pixel-mutating step from a dangerous one.
