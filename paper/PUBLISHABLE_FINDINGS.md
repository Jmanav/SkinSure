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

## F1 — The ensemble configuration cannot be resolved on melanoma sensitivity at this sample size; keep equal weights

**Status: CONFIRMED (as a null result, which is the finding). Tested 2026-11-02.**
**Basis: clean 565 + contaminated 727 + validation 748. Source: `.nb1out/ensemble_weighting_analysis.md` and the `weighting_*.csv` files.**

> **Resolved.** The provisional claim below — that the shipped ECS ensemble is
> meaningfully worse on melanoma sensitivity — **did not survive testing.** The
> difference is 2 discordant images (McNemar exact p = 0.50), all seven configs'
> Wilson intervals mutually overlap, and the sample cannot resolve an effect this
> small: pooling test + validation to 326 melanoma cases gives 5 unanimously
> ES-favouring discordances, and 5 one-sided pairs cannot produce p < 0.0625 no
> matter how consistent they are. α=0.05 would need ≥6.
>
> Meanwhile the better-powered statistics favour ECS: it wins macro AUC on both
> bases (0.98132 vs 0.97722 test), and switching to ES costs ~2 pp accuracy
> *significantly* (test p = 0.043, val p = 0.020) and is worse on four of five
> validation classes including BCC (−3.36 pp). **Paying a measured cost for an
> unmeasured benefit is the wrong trade. Keep equal weights.**
>
> Honest residual risk: the ~1.5 pp effect may be real and one-sided across all
> three bases, i.e. ECS may miss roughly 1 melanoma in 65 that ES catches.
> Settling it needs ~1,500–2,000 melanoma images for 80% power at 1.5 pp; we have
> 124, or 326 pooled. If weights are ever tuned, tune on validation and report on
> test — never the test-set grid search, which is optimistically biased by
> construction (28.3% of 1,033 weight vectors hit the same best melanoma count,
> so the choice is mostly noise).
>
> **Publishable as a null result with a power analysis** — a useful corrective to
> ablation tables that rank configurations on fractions of a percent without
> testing whether the ranking is distinguishable. See also F4, the concrete
> safety gap found while testing this.

### The original provisional claim, for the record

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

## F4 — The melanoma escalation rule is structurally unreachable for every melanoma the ensemble misses

**Status: CONFIRMED (verified directly against the code and the misclassified cases).**
**Logged 2026-11-02. Basis: clean 565. Source: `.nb1out/weighting_melanoma_misses.csv`, `src/agents/severity_triage.py`.**

`severity_triage.py` Rule 2 escalates to `urgent` when
`predictions.primary_label == MELANOMA_LABEL and primary_confidence >= 0.40`.
It keys on **`primary_label`** — the argmax. A melanoma that ranks *second* can
therefore never reach the rule, whatever the threshold is set to.

All three melanomas the shipped ensemble misses are exactly that shape:

| Image | ECS prediction | p(melanoma) |
|---|---|---|
| ISIC_6966105 | Basal_Cell_Carcinoma | 0.343 |
| ISIC_7619041 | Keratosis | 0.369 |
| ISIC_7406359 | Keratosis | 0.064 |

Two of the three carry p(melanoma) ≈ 0.34–0.37 — ranked high, not top. Lowering
`MELANOMA_URGENT_MIN_CONFIDENCE` from 0.40 would **not** catch them, because the
rule never fires: the gate is the argmax, not the threshold. This is a
structural gap, not a tuning problem, and it is invisible to any accuracy or
sensitivity metric computed on argmax predictions.

**The cheap fix, measured:** escalating on `p(melanoma) ≥ 0.30` *wherever
melanoma ranks* lifts ECS from 121/124 to 123/124 (97.58% → 99.19%) — exactly
matching the best configuration tested — at a cost of **one** extra flag on a
non-melanoma image across 441 (3 → 4), with no accuracy and no BCC loss. That
is the same sensitivity the ensemble-weighting change appeared to offer, with no
accuracy trade and no change to the model.

**Not implemented** — this is a triage-rule proposal awaiting review, since it
changes clinical escalation behaviour. The 0.30 threshold is itself fitted to
these three cases and should be chosen on validation before shipping.

**Why it is publishable:** a safety rule keyed on argmax is a plausible and
easily-missed failure mode for any multi-class clinical classifier with a
privileged dangerous class. The metric that should have caught it — melanoma
sensitivity — cannot, because it is computed on the same argmax the rule gates
on.

---

## F5 — A standard dermoscopy preprocessing step (DullRazor hair removal) measurably harms a model trained without it, and damages the class it fires on

**Status: CONFIRMED (full paired A/B over the clean 565 set).**
**Logged 2026-11-02. Source: `.nb1out/hair_ab_raw.csv`, `scripts/hair_ab_experiment.py`, `scripts/hair_ab_analyze.py`.**

DullRazor-style black-hat hair removal is routine in dermoscopy pipelines. Run
as an inference-time step in front of models trained on resize + ImageNet
normalisation only, it **never corrected a single image out of 565 and broke
two** — both true Basal Cell Carcinoma, i.e. two new false negatives on a
malignancy:

| Image | A → B | BCC confidence |
|---|---|---|
| ISIC_0054063 | BCC → **Keratosis** (benign) | 0.9314 → 0.3366 |
| ISIC_0073116 | BCC → Melanoma | 0.6303 → 0.3881 |

The first is the worst outcome available: a confident malignant call turned
benign.

| Metric | Baseline | Hair removal | Δ |
|---|---|---|---|
| Accuracy | 87.61% | 87.26% | −0.35 pp |
| Macro AUC | 0.9813 | 0.9802 | −0.0011 |
| **Melanoma sens** | 97.58% | 97.58% | **0.00** |
| **BCC sens** | 100% | 98.11% | **−1.89 pp** |

On the 41 images where the step fired: accuracy −4.88 pp, BCC sensitivity
−7.69 pp. McNemar b=2, c=0, exact p=0.50 — not significant on two discordant
pairs, but the sign is unanimous: nothing was ever fixed.

**The mechanism is the publishable part.** The step is not merely unhelpful, it
is mistuned in a way that selectively damages malignancy. At `threshold=10` the
black-hat filter is non-selective on this dataset: across the 524 images it
declined, the mask covered a **median 58.9%** of the frame and ≥75% on 185 of
them. A `MAX_MASK_FRACTION` guard at 10% is the only thing preventing wholesale
inpainting over lesions, and the 7.26% that slip under it inpainted a median
6.9% of pixels — sitting just below the cutoff rather than being precise hair
masks. Firing therefore concentrates on BCC (24.5% vs 1.5–5.7% for other
classes), because BCC lesions are dark and textured and respond to black-hat
much as hair does. **The step preferentially fires on, and damages, the class it
is most dangerous to damage.**

**Methodological note worth reporting alongside it:** a control check on the 524
declined images found zero metric change and a maximum probability shift of
0.00000, confirming the scoring path is deterministic and that both label flips
are genuine inpainting effects rather than noise. Stratifying by whether a
conditional preprocessing step actually fired is essential — a whole-set delta
dilutes a −4.88 pp effect to −0.35 pp and would have been dismissed as noise.

**Open:** artifact removal (reported 47% firing) has had no equivalent test.

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
