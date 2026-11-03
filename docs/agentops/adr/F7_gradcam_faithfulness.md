# ADR-F7: Grad-CAM is measurably faithful; SHAP comparison deferred

**Status:** Accepted  
**Date:** 2026-11-03  
**Decision-makers:** Manav Jain  
**Evidence:** `.nb1out/xai_aopc_analysis.md`, `.nb1out/xai_aopc_raw.csv`, `paper/experiments/07_xai_aopc_gradcam_vs_random.py`, `scripts/xai_aopc_experiment.py`, `scripts/evaluate_xai.py`

## Context

Grad-CAM was chosen for the `/predict` path over SHAP on latency and output-shape grounds (~5 s vs minutes per image; dense 7×7 spatial map vs per-patch attributions). But faithfulness — whether the heatmaps identify pixels the model actually uses, rather than producing plausible-looking but decorative overlays — had never been measured. If the maps were random noise the entire `XAISummary` bridge (focus region, boundary alignment, artifact flag) would be meaningless, and the clinical summarizer would be reasoning over fiction.

## Decision

Grad-CAM is measurably faithful on these models: it beats a structure-matched random baseline by a large, statistically significant margin on AOPC. The SHAP comparison is **deferred by decision**, not pending or blocked. It would be the experiment that could overturn the `/predict` method choice, but nothing in the product waits on it: Grad-CAM has a positive faithfulness floor, SHAP is not installed, and `/explain` (where SHAP would serve) does not exist. Trigger to reopen: `/explain` gets scheduled, or a paper reviewer asks for the comparison.

## Evidence summary

**Method:** AOPC (Area Over the Perturbation Curve) — progressively delete the top-k% most-attributed pixels and measure how much p(predicted class) drops. A faithful map causes a steeper drop than a random one. Run on a Kaggle T4 GPU.

**Sample:** 150 images from the de-duplicated **565-image clean test set**, 30 per class, all five classes.

| Method | Mean AOPC | 95% CI (bootstrap) |
|---|---|---|
| Grad-CAM | 0.3099 | [0.2831, 0.3376] |
| random7x7 (structure-matched) | 0.1834 | [0.1678, 0.1994] |
| randompixel (sanity check) | 0.4887 | [0.4576, 0.5196] |

**Paired comparison (Grad-CAM − random7x7):** mean diff **+0.1266** [+0.1059, +0.1478], Wilcoxon W = 747, **p < 0.0001**, **127/150 wins (84.7%)**, Cohen's dz = **0.967**. Significant in all five per-class rows (p ≤ 0.0012). Holds on images the ensemble got **wrong** (+0.1479, p = 0.0006) — faithfulness is a property of the explanation, not of being correct.

**The `randompixel` trap:** i.i.d. pixel-level speckle deletion posts the *highest* AOPC (0.4887) while having zero localisation content. It shreds texture globally and pushes the image off-distribution, measuring model fragility to high-frequency corruption rather than explanation quality. It is a sanity check that AOPC is OOD-sensitive, never a rival method. Only the `random7x7` comparison (matched spatial structure) is the result.

**Bug found and fixed during analysis:** the script used raw `(1,1,1)` weights but `SkinEnsemble` normalises them, so every probability was 3× too large. Verified harmless — AOPC is a linear functional of the deletion curve, so the constant factor rescaled every value identically. Win count and dz are bit-identical before and after; only magnitudes changed. The script now normalises weights and asserts probabilities sum to 1.

## Alternatives considered

1. **Run the SHAP comparison now.** SHAP is not installed, the teammate's notebooks used different blanking baselines in each, and `/explain` does not exist. The cost of integrating and re-scoring under one harness (`scripts/evaluate_xai.py`) is real and buys nothing the product currently needs.
2. **Treat Grad-CAM as unvalidated and remove XAISummary from the clinical path.** Disproportionate — the AOPC result shows the maps track model-relevant pixels, so the bridge's issues (gap 7: `boundary_alignment` never reaches `good`) are threshold problems, not signal problems.
3. **Acquire annotated lesion masks to test whether Grad-CAM highlights *the lesion*.** Out of scope (decided 2026-11-03). AOPC measures whether the map found pixels the model *used*, which is the faithfulness question. Whether those pixels are the lesion is a different question requiring ground-truth masks this project does not have and will not acquire.

## Consequences

- The `XAISummary` bridge stays in the clinical path. Its fields (`focus_description`, `boundary_alignment`) are **advisory, never evidence** — the faithfulness result locates the fault in the bridge's thresholds (gap 7), not in the heatmaps.
- The heatmap → summary thresholds should be re-tuned: a field that never reaches its best value across all five classes is self-evidently mis-cut. This is the tractable follow-up.
- If `/explain` is ever scheduled or a reviewer asks, re-score the `xai-shap` branch's maps with `scripts/evaluate_xai.py` so both methods are measured under one identical AOPC procedure, then decide whether SHAP earns its integration cost.
- The `artifact_flag` shares the bridge whose thresholds are suspect. Evidence so far is reassuring (fired 0/15 on real photos) but thin. Production monitoring of the recapture-loop rate (A1 tracing) is the check that catches a mis-tuned flag without needing masks.
