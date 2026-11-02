# Publishable findings

Results flagged as paper material. Each entry states its evidential status
plainly: **PROVISIONAL** means the effect is observed but not yet tested for
significance and must not be published as stated; **CONFIRMED** means a
significance test has been run and is recorded here.

Every figure must name its test set. Three are in play and are **not**
comparable with each other:

| Basis | n | What it is |
|---|---|---|
| FL split | 997 | Federated-learning held-out set (`fedler.md`). Parked 2026-11-02, pending re-run on a different dataset. |
| Paper, contaminated | 727 | The original test split. |
| **Paper, clean** | **565** | The 727 after pHash de-duplication removed 162 (22.3%). **The primary reporting basis.** |

---

## F1 — The shipped 3-model ensemble has the worst melanoma sensitivity of any configuration tested

**Status: PROVISIONAL — significance not yet tested. Do not publish as stated.**
**Logged 2026-11-02. Basis: clean 565. Source: `.nb1out/per_class_metrics.csv`, `.nb1out/ablation_table.csv` (NB1, `paper/experiments/01_eval_and_ablation.ipynb`).**

The deployed configuration is an equal-weight (1.0/1.0/1.0) mean of three
backbones — EfficientNet-B1, ConvNeXt-Tiny, Swin-Tiny — set in
`configs/models.yaml`. All figures below are on the v2 retrained checkpoints.

| Config | Accuracy | Melanoma sens | BCC sens | Macro AUC |
|---|---|---|---|---|
| ConvNeXt alone | **88.67%** | 97.58% | 100% | 0.9641 |
| **ECS — shipped** | 87.79% | **97.58%** | 100% | **0.9813** |
| ConvNeXt + Swin | 88.32% | 98.39% | 100% | 0.9766 |
| EfficientNet + Swin | 85.84% | **99.19%** | 100% | 0.9772 |

The shipped ECS configuration is tied for the **lowest** melanoma sensitivity
of the seven configurations evaluated. Dropping ConvNeXt (the ES pair) gains
1.6 pp of melanoma sensitivity at a cost of 2.0 pp of accuracy.

**Why this is the interesting direction.** For a melanoma screening tool a
missed melanoma is a materially worse outcome than a misclassified keratosis,
so sensitivity is the quantity to protect and accuracy the one to trade. The
shipped default currently trades the other way, and it does so by accident —
equal weighting was never selected against a sensitivity criterion.

**The caveat that blocks publication.** At 124 melanoma cases in the clean set,
97.58% vs 99.19% is **3 missed cases vs 1**. Two patients is a real difference
and not nothing, but it sits well inside sampling noise at this n. Wilson 95%
CIs on those two proportions will very likely overlap substantially. Nothing
here may be written up as an effect until that is computed.

**What would move this to CONFIRMED:**
1. Wilson 95% CIs on melanoma sensitivity for all seven configurations.
2. McNemar's test on ECS-vs-ES melanoma outcomes — paired, same images, which
   is the correct test for two conditions over one sample and has far more
   power than comparing two independent proportions.
3. Whether tuning the three weights beats dropping a backbone outright.
4. Whether the ranking survives on the contaminated 727 basis, as a robustness
   check.

If the CIs overlap and McNemar returns non-significant, the honest finding is
"no measurable difference in melanoma sensitivity between configurations at
n=565, and the choice should rest on AUC or on a larger validation set" — which
is itself publishable, and is the more likely outcome.

**Related, same source, already solid:** ConvNeXt v2 is the strongest single
backbone (88.67% clean, 89.13% on 727), reversing the v1 finding where it was
the weakest at 71.35%. That reversal is attributable to a training-schedule fix
(`freeze_epochs` 8→3, `patience` 5→10, `num_epochs` 30→40, seed unchanged), not
to architecture — see `paper/new_results.md`.

---

## F2 — pHash de-duplication removes 22.3% of the test set; measured inflation is small

**Status: CONFIRMED (descriptive — a measured count, not an inferential claim).**
**Logged 2026-11-02. Source: `.nb1out/results.json`, `.nb1out/contamination_comparison.csv`.**

`02_dedup_phash.ipynb` found **162 of 727 test images (22.28%)** with a
perceptual-hash near-duplicate (Hamming d ≤ 5) in train or val. The clean
manifest keeps **565**: Keratosis 133, Melanoma 124, Eczema 111, BCC 106,
Tinea 91.

Accuracy inflation from the contamination is **0.31–0.81 pp** across the seven
configurations — real, consistently positive, and smaller than one might expect
from a 22% duplicate rate. Worth reporting precisely because the honest
magnitude is modest; papers that de-duplicate rarely publish how little it
moved.

---

## F3 — Prompt-context defect: a derived figure needs its denominator

**Status: CONFIRMED (reproduced in two independent live runs, then fixed).**
**Logged 2026-10-31. Source: live Claude Haiku 5.5 calls via `scripts/summarize_one.py`.**

`build_context()` passed the ensemble's `agreement_score` as a bare fraction
("Model agreement: 0.33"). In both live runs that exposed it, the LLM inferred
the wrong number of models — "the two models agree at only 0.33" when 1 of 3
agreed, and "the two models agree completely" when 3 of 3 did. After the
context line was changed to state the count with its denominator ("1 of 3
models agree with the primary label (score 0.33, low tier)"), the next run
reported "All three models agree" correctly.

Small, but a clean and reproducible instance of a general lesson for
LLM-in-the-loop clinical systems: **a derived number handed to a model needs its
denominator, or the model invents one** — and the invented value lands in
patient-facing text. n is tiny (2 wrong before, 1 right after); report it as an
observation and a design rule, not as a measured rate.

---

## Methodological notes worth a paragraph in the paper

- **`agreement_tiers.csv` is mis-binned and must not be quoted.** NB1 compared a
  float agreement score against the literals `0.33`/`0.67`, but the score can
  only be 0, 1/3, 2/3 or 1, and `1/3 = 0.3333… ≠ 0.33`, so both mixed tiers
  reported n=0 and 81 of 565 images vanished from the table.
  `.nb1out/_agree.py` re-derives the correct tiers. The shipped code is
  unaffected — `Predictions.confidence_tier()` uses `>=` thresholds, not
  equality.
- **Grad-CAM is used at inference for latency and output shape, not because it
  was shown to be more faithful than SHAP.** That comparison has never been run
  (`scripts/evaluate_xai.py` is the harness). Any xAI claim in the paper must
  not imply a faithfulness result that does not exist.
- **The heatmap → `XAISummary` bridge is uncalibrated.** Over 15 real photos
  `boundary_alignment` was `poor` 11 times, `partial` 2, `unknown` 2 and
  **never `good`**. A field that never attains its best value across a full
  class spread is more likely mis-thresholded than correct.
