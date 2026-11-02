# Should the 3-backbone ensemble keep equal weights? — a statistical answer

Analysis date 2026-11-02. Branch `federated-learning`. **Analysis only — no config was changed.**
Evidence: cached per-backbone probabilities in `.nb1out/cache_{efficientnet_b1,convnext_tiny,swin_tiny}_{test,val}.npz`.
No inference was re-run. Reproducibility script: `.nb1out/_weighting_analysis.py`.

---

## Verdict

**Keep the equal-weight 3-backbone configuration. The ECS-vs-ES melanoma-sensitivity
difference is not statistically distinguishable at n=565 — and it is not distinguishable
even after pooling test and validation into 326 melanoma cases.**

The headline gap in `per_class_metrics.csv` — ECS 97.58% vs ES 99.19% melanoma sensitivity —
rests on **three images**, and on the clean 565 only **two** of them are discordant
(ECS wrong where ES is right; there are **zero** cases the other way). McNemar on melanoma
cases gives b=0, c=2, exact two-sided **p = 0.50**. The Wilson 95% CIs are
ECS [93.13%, 99.17%] and ES [95.57%, 99.86%] — they overlap across a 3.60 pp span, and in
fact all seven configurations' intervals mutually overlap. Nothing in this test set separates
any config from any other on melanoma sensitivity.

Three further findings turn "not significant" into a positive case for keeping ECS rather
than a shrug:

1. **The direction is consistent but the sample is underpowered by construction.** ES beats
   ECS on melanoma on all three bases (clean 565: +1.61 pp; contaminated 727: +0.53 pp;
   validation 748: +1.49 pp), and all 5 pooled discordant pairs favour ES — a perfectly
   one-sided result. Pooled exact two-sided p is still only **0.0625**, because 5 unanimous
   discordances is the *best p this many pairs can produce*. Reaching α=0.05 requires at
   least **6** consistently-signed discordant melanoma cases; we have 5 out of 326 melanoma
   images. So the honest reading is: there may be a real ~1.5 pp effect, and this dataset
   cannot resolve it.

2. **The better-powered statistics point the opposite way.** Macro AUC is threshold-free and
   uses all 565 (and all 748) images instead of 2–3 discordant melanomas. **ECS wins it on
   both bases**: 0.98132 vs 0.97722 on test (−0.00411 for ES), 0.97822 vs 0.97371 on
   validation (−0.00451). Melanoma-vs-rest AUC also favours ECS on both
   (0.99872 vs 0.99664 test; 0.99679 vs 0.99655 val). ECS's *ranking* of melanoma risk is
   better even where its argmax occasionally loses.

3. **Switching to ES has a real, statistically supported cost.** ECS→ES loses ~2 pp of
   accuracy on both bases, and unlike the melanoma comparison this one *is* significant:
   test b=18, c=7, exact p = **0.0433**; validation b=26, c=11, exact p = **0.0201**. On
   validation the drop is not confined to accuracy — ES is *worse* on four of five classes
   including BCC (−3.36 pp), the secondary safety target. Trading a statistically significant
   2 pp accuracy loss and a BCC regression for a melanoma gain that cannot be measured is a
   bad exchange.

### Residual risk, stated plainly

The risk I am accepting by recommending "keep ECS" is that the ~1.5 pp melanoma effect is
real and ECS misses roughly 1 melanoma in 65 that ES would catch. That is not a comfortable
risk in a screening tool, and the one-sided consistency across three bases is genuinely
suggestive rather than nothing. I am recommending against acting on it because (a) acting
means paying a *measured* cost for an *unmeasured* benefit, and (b) there is a cheaper lever
that buys the same sensitivity without the trade — see below.

### The cheaper lever (recommended follow-up, not a config change)

All three melanomas ECS misses are cases where ECS ranks melanoma high but not highest:
p(melanoma) = 0.3427, 0.3690 and 0.0645. Escalating on **p(melanoma) ≥ 0.30** rather than
requiring argmax recovers 2 of the 3 and lifts ECS melanoma sensitivity to **99.19%
(123/124) — matching ES exactly — at a cost of 4 melanoma flags on non-melanoma images out
of 441 (0.9%), versus 3 under the current argmax behaviour.** That is **one** extra
precautionary flag across 441 images to recover the melanoma ES catches, with **no** loss of
accuracy, no loss of BCC sensitivity, and no weight change. It also fits the existing
architecture: `severity_triage.py` already owns threshold-based escalation ("LLM interprets;
Python decides"), so this is a triage-rule question, not an ensemble-weights question.

**This lever is confirmed *not* in place today.** `src/agents/severity_triage.py` Rule 2 reads
`predictions.primary_label == MELANOMA_LABEL and predictions.primary_confidence >= 0.40`
(`MELANOMA_URGENT_MIN_CONFIDENCE = 0.40`). Because it gates on `primary_label` — the argmax —
a melanoma that the ensemble ranks *second* at p=0.34 never reaches the rule at all, no
matter how the 0.40 threshold is set. All three missed melanomas are exactly that shape:
melanoma is not the argmax, so Rule 2 is unreachable for them. Making the rule consider
melanoma's probability wherever it ranks (e.g. escalate when `p(melanoma) >= 0.30` even as
the secondary label) is the change that would capture the sensitivity ES appears to offer.
That is a triage-rule proposal for the team, out of scope for this analysis — no code or
config was modified.

The third miss (ISIC_7406359.jpg, p(melanoma) = 0.0645) is missed by every configuration
including ES — no weighting recovers it.

### What would settle the weighting question

- **More melanoma cases.** To detect a ~1.5 pp sensitivity difference at 80% power you need
  on the order of 1,500–2,000 melanoma images, not 124. Pooling test+val got to 326 and
  p=0.0625; roughly 5× more melanoma data would be needed to resolve it.
- **A proper weight-selection protocol.** If weights are ever tuned, tune them on validation
  and report on test. The grid search in §3 is fitted on test and is therefore optimistically
  biased (see the warning there) — it establishes only that headroom exists, not what to ship.
- **Decision-curve / cost-weighted analysis** rather than raw sensitivity, since the real
  quantity of interest is "missed melanomas per unnecessary referral", which §6's threshold
  table begins to expose.

### Answer to the grid-search sub-question

Does a non-equal weighting of all three backbones beat dropping one? **On melanoma
sensitivity, it ties rather than beats:** the best all-weights-nonzero vector reaches
99.19% (123/124), exactly the same as ES, but keeps macro AUC at 0.98103 (vs ES's 0.97722)
and accuracy at 87.96% (vs ES's 85.84%). So *if* one wanted ES's melanoma sensitivity, a
reweighted three-model ensemble is strictly the better way to get it than dropping ConvNeXt.
But this is a test-set-fitted observation (§3), the sensitivity difference it chases is the
same unmeasurable one, and 28.3% of the entire 1,033-vector grid reaches 123/124 — which is
itself evidence that which weights you pick is mostly noise at this sample size. Equal
weights sit comfortably inside that noise band and require no justification to a reviewer.

### Supporting files

| File | Contents |
|---|---|
| `.nb1out/weighting_sanity_check.csv` | Reconstruction vs `ablation_table.csv`, all 7 configs |
| `.nb1out/weighting_wilson_ci.csv` | Melanoma and BCC sensitivity with Wilson 95% CIs |
| `.nb1out/weighting_mcnemar.csv` | All McNemar comparisons, discordant counts, exact p-values |
| `.nb1out/weighting_grid_search.csv` | All 1,033 weight vectors × 7 metrics (test-set fitted) |
| `.nb1out/weighting_robustness.csv` | Contaminated-727 and validation-748 replication |
| `.nb1out/weighting_melanoma_misses.csv` | The 3 missed melanomas with per-backbone probabilities |
| `.nb1out/_weighting_analysis.py` | The analysis script (reruns everything above) |

---

# Full analysis

## Method notes

- Configs are reconstructed exactly as NB1 does: the **equal-weight mean of the member
  backbones' probability vectors, argmax for the prediction**. Clean-565 rows are selected
  from the 727 cached rows by matching `parent-dir name / filename` against
  `C:\Users\Cipher\Downloads\test_clean_manifest.csv`, the convention in `.nb1out/_agree.py`.
- The three caches are asserted row-aligned on identical `paths` and `labels`, and each
  cache's `classes` array is asserted equal to
  `[Basal_Cell_Carcinoma, Eczema, Keratosis, Melanoma, Tinea]`. Melanoma is index 3.
- **`scipy` is not installed in `venv/`.** The Wilson interval, the exact two-sided binomial
  test, the McNemar χ² (corrected and uncorrected), the χ²(1) survival function
  (`erfc(sqrt(x/2))`), ROC AUC (rank / Mann-Whitney identity with ties averaged) and
  multiclass MCC are implemented directly in the script. The exact binomial is the quoted
  p-value throughout, since b+c is small in every melanoma comparison; with p=0.5 the
  binomial is symmetric, so two-sided = 2 × the smaller tail, capped at 1.
- `Melanoma` support is 124 on the clean 565, matching `per_class_metrics.csv`.

Clean test set: n=565 (expected 565 of 727)
Class counts: Basal_Cell_Carcinoma=106, Eczema=111, Keratosis=133, Melanoma=124, Tinea=91

## 0. Reproduction check against ablation_table.csv

| config | acc (mine) | acc (NB1) | delta | macroAUC (mine) | macroAUC (NB1) | delta | MCC (mine) | MCC (NB1) | delta |
|---|---|---|---|---|---|---|---|---|---|
| E | 0.842478 | 0.842478 | +0.00e+00 | 0.967791 | 0.967791 | +0.00e+00 | 0.806577 | 0.806577 | +0.00e+00 |
| C | 0.886726 | 0.886726 | +0.00e+00 | 0.964106 | 0.964106 | +0.00e+00 | 0.860325 | 0.860325 | -1.11e-16 |
| S | 0.856637 | 0.856637 | +0.00e+00 | 0.970683 | 0.970683 | +2.22e-16 | 0.823145 | 0.823145 | +0.00e+00 |
| EC | 0.881416 | 0.881416 | +0.00e+00 | 0.979762 | 0.979762 | +0.00e+00 | 0.854535 | 0.854535 | +0.00e+00 |
| ES | 0.858407 | 0.858407 | +0.00e+00 | 0.977216 | 0.977216 | +0.00e+00 | 0.825547 | 0.825547 | +0.00e+00 |
| CS | 0.883186 | 0.883186 | +0.00e+00 | 0.976623 | 0.976623 | +2.22e-16 | 0.856699 | 0.856699 | +0.00e+00 |
| ECS | 0.877876 | 0.877876 | +0.00e+00 | 0.981325 | 0.981325 | +0.00e+00 | 0.850445 | 0.850445 | -1.11e-16 |

Max absolute deviation across all 21 cells: **2.22e-16**. Reconstruction matches NB1 exactly; everything downstream is trustworthy.

## 1. Wilson 95% CIs on melanoma sensitivity (clean 565)

Melanoma cases: n=124

| config | TP | FN | sensitivity | Wilson 95% CI | CI width |
|---|---|---|---|---|---|
| E | 122 | 2 | 98.39% | [94.31%, 99.56%] | 5.25 pp |
| C | 121 | 3 | 97.58% | [93.13%, 99.17%] | 6.05 pp |
| S | 122 | 2 | 98.39% | [94.31%, 99.56%] | 5.25 pp |
| EC | 121 | 3 | 97.58% | [93.13%, 99.17%] | 6.05 pp |
| ES | 123 | 1 | 99.19% | [95.57%, 99.86%] | 4.28 pp |
| CS | 122 | 2 | 98.39% | [94.31%, 99.56%] | 5.25 pp |
| ECS | 121 | 3 | 97.58% | [93.13%, 99.17%] | 6.05 pp |

Secondary target -- BCC sensitivity (n=106):

| config | TP | FN | sensitivity | Wilson 95% CI |
|---|---|---|---|---|
| E | 105 | 1 | 99.06% | [94.85%, 99.83%] |
| C | 106 | 0 | 100.00% | [96.50%, 100.00%] |
| S | 104 | 2 | 98.11% | [93.38%, 99.48%] |
| EC | 106 | 0 | 100.00% | [96.50%, 100.00%] |
| ES | 106 | 0 | 100.00% | [96.50%, 100.00%] |
| CS | 106 | 0 | 100.00% | [96.50%, 100.00%] |
| ECS | 106 | 0 | 100.00% | [96.50%, 100.00%] |

**Overlap:** ECS CI [93.13%, 99.17%] and ES CI [95.57%, 99.86%] overlap over [95.57%, 99.17%] -- a span of 3.60 pp.

In fact every one of the seven intervals overlaps every other: the lowest upper bound is 99.17% and the highest lower bound is 95.57%, so no pair of configurations is separated on melanoma sensitivity.

## 2. McNemar's test, ECS vs ES

Convention: b = ECS correct & ES wrong; c = ECS wrong & ES correct.

### Melanoma cases only

n = 124 paired images
```
                       ES correct   ES wrong
  ECS correct          121                0   <- b
  ECS wrong              2                1
                            ^ c
```
- Discordant pairs: b=0 (ECS right, ES wrong), c=2 (ECS wrong, ES right), b+c=2
- McNemar chi2 (continuity-corrected) = 0.5000, 1 df, asymptotic p = 0.4795
- McNemar chi2 (uncorrected) = 2.0000, asymptotic p = 0.1573
- **Exact two-sided binomial p = 0.5000**  (b+c small, so the exact test is the one to quote)

### All 565 images (overall accuracy)

n = 565 paired images
```
                       ES correct   ES wrong
  ECS correct          478               18   <- b
  ECS wrong              7               62
                            ^ c
```
- Discordant pairs: b=18 (ECS right, ES wrong), c=7 (ECS wrong, ES right), b+c=25
- McNemar chi2 (continuity-corrected) = 4.0000, 1 df, asymptotic p = 0.0455
- McNemar chi2 (uncorrected) = 4.8400, asymptotic p = 0.0278
- **Exact two-sided binomial p = 0.0433**

### Supporting comparisons (same protocol)

### Melanoma only, ECS vs CS

n = 124 paired images
```
                       CS correct   CS wrong
  ECS correct          121                0   <- b
  ECS wrong              1                2
                            ^ c
```
- Discordant pairs: b=0 (ECS right, CS wrong), c=1 (ECS wrong, CS right), b+c=1
- McNemar chi2 (continuity-corrected) = 0.0000, 1 df, asymptotic p = 1.0000
- McNemar chi2 (uncorrected) = 1.0000, asymptotic p = 0.3173
- **Exact two-sided binomial p = 1.0000**  (b+c small, so the exact test is the one to quote)

### Melanoma only, ECS vs C

n = 124 paired images
```
                       C correct   C wrong
  ECS correct          121                0   <- b
  ECS wrong              0                3
                            ^ c
```
- Discordant pairs: b=0 (ECS right, C wrong), c=0 (ECS wrong, C right), b+c=0
- McNemar chi2 (continuity-corrected) = 0.0000, 1 df, asymptotic p = 1.0000
- McNemar chi2 (uncorrected) = 0.0000, asymptotic p = 1.0000
- **Exact two-sided binomial p = 1.0000**  (b+c small, so the exact test is the one to quote)

### Melanoma only, ES vs CS

n = 124 paired images
```
                       CS correct   CS wrong
  ES correct          122                1   <- b
  ES wrong              0                1
                            ^ c
```
- Discordant pairs: b=1 (ES right, CS wrong), c=0 (ES wrong, CS right), b+c=1
- McNemar chi2 (continuity-corrected) = 0.0000, 1 df, asymptotic p = 1.0000
- McNemar chi2 (uncorrected) = 1.0000, asymptotic p = 0.3173
- **Exact two-sided binomial p = 1.0000**  (b+c small, so the exact test is the one to quote)


## 3. Weight grid search (FITTED ON THE TEST SET -- optimistically biased)

> **Read this before using any number in this section.** The grid below is scored on the same 565 images it selects on. Whatever it picks is the maximum of ~230 correlated estimates, so its apparent score is biased upward by the selection itself. It answers *is there headroom?*, not *what weights should we ship?*. A weight vector taken from here and deployed would be expected to regress toward the equal-weight result on new data.

Grid: each weight in 0.0..1.0 step 0.1, normalised to sum 1, all-zero skipped, duplicates after normalisation removed -> **1033 distinct weight vectors**.


Equal-weight ECS reference row: mel sens 97.58% (121/124), acc 87.79%, macro AUC 0.98132, MCC 0.8504.

### (a-c) Top 10 by mel_sens

| rank | w_E | w_C | w_S | mel sens | mel TP/124 | BCC sens | accuracy | macro AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.333 | 0.250 | 0.417 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98103 | 0.8519 |
| 2 | 0.350 | 0.250 | 0.400 | 99.19% | 123/124 | 100.00% | 87.79% | 0.98099 | 0.8498 |
| 3 | 0.304 | 0.261 | 0.435 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98099 | 0.8519 |
| 4 | 0.267 | 0.267 | 0.467 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98099 | 0.8519 |
| 5 | 0.273 | 0.273 | 0.455 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98098 | 0.8519 |
| 6 | 0.316 | 0.263 | 0.421 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98097 | 0.8519 |
| 7 | 0.263 | 0.263 | 0.474 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98097 | 0.8519 |
| 8 | 0.375 | 0.250 | 0.375 | 99.19% | 123/124 | 100.00% | 87.61% | 0.98096 | 0.8475 |
| 9 | 0.360 | 0.240 | 0.400 | 99.19% | 123/124 | 100.00% | 87.61% | 0.98095 | 0.8475 |
| 10 | 0.300 | 0.250 | 0.450 | 99.19% | 123/124 | 100.00% | 87.96% | 0.98094 | 0.8519 |

Best mel_sens = 0.991935 at weights (E=0.333, C=0.250, S=0.417); equal weights give 0.975806 (gap +0.016129).

### (a-c) Top 10 by macro_auc

| rank | w_E | w_C | w_S | mel sens | mel TP/124 | BCC sens | accuracy | macro AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.286 | 0.429 | 0.286 | 97.58% | 121/124 | 100.00% | 88.32% | 0.98163 | 0.8566 |
| 2 | 0.278 | 0.444 | 0.278 | 97.58% | 121/124 | 100.00% | 88.50% | 0.98162 | 0.8589 |
| 3 | 0.292 | 0.417 | 0.292 | 97.58% | 121/124 | 100.00% | 88.14% | 0.98162 | 0.8544 |
| 4 | 0.235 | 0.471 | 0.294 | 97.58% | 121/124 | 100.00% | 88.32% | 0.98162 | 0.8566 |
| 5 | 0.238 | 0.476 | 0.286 | 97.58% | 121/124 | 100.00% | 88.50% | 0.98161 | 0.8586 |
| 6 | 0.263 | 0.474 | 0.263 | 97.58% | 121/124 | 100.00% | 88.32% | 0.98161 | 0.8566 |
| 7 | 0.267 | 0.467 | 0.267 | 97.58% | 121/124 | 100.00% | 88.32% | 0.98161 | 0.8566 |
| 8 | 0.294 | 0.412 | 0.294 | 97.58% | 121/124 | 100.00% | 88.14% | 0.98161 | 0.8544 |
| 9 | 0.273 | 0.455 | 0.273 | 97.58% | 121/124 | 100.00% | 88.50% | 0.98161 | 0.8589 |
| 10 | 0.250 | 0.450 | 0.300 | 97.58% | 121/124 | 100.00% | 88.50% | 0.98160 | 0.8589 |

Best macro_auc = 0.981629 at weights (E=0.286, C=0.429, S=0.286); equal weights give 0.981325 (gap +0.000304).

### (a-c) Top 10 by accuracy

| rank | w_E | w_C | w_S | mel sens | mel TP/124 | BCC sens | accuracy | macro AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.357 | 0.643 | 0.000 | 97.58% | 121/124 | 100.00% | 89.20% | 0.97996 | 0.8670 |
| 2 | 0.333 | 0.667 | 0.000 | 97.58% | 121/124 | 100.00% | 89.20% | 0.97989 | 0.8670 |
| 3 | 0.214 | 0.643 | 0.143 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98124 | 0.8650 |
| 4 | 0.333 | 0.500 | 0.167 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98122 | 0.8648 |
| 5 | 0.333 | 0.533 | 0.133 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98115 | 0.8650 |
| 6 | 0.200 | 0.667 | 0.133 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98115 | 0.8650 |
| 7 | 0.308 | 0.538 | 0.154 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98112 | 0.8650 |
| 8 | 0.333 | 0.556 | 0.111 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98108 | 0.8650 |
| 9 | 0.312 | 0.562 | 0.125 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98107 | 0.8650 |
| 10 | 0.154 | 0.692 | 0.154 | 97.58% | 121/124 | 100.00% | 89.03% | 0.98107 | 0.8650 |

Best accuracy = 0.892035 at weights (E=0.357, C=0.643, S=0.000); equal weights give 0.877876 (gap +0.014159).

### Distribution of melanoma sensitivity across the grid

| melanoma TP | sensitivity | # weight vectors | share of grid |
|---|---|---|---|
| 123/124 | 99.19% | 292 | 28.3% |
| 122/124 | 98.39% | 209 | 20.2% |
| 121/124 | 97.58% | 532 | 51.5% |

### Does a genuinely 3-backbone weighting (all weights > 0) beat dropping one?

- Best all-three-nonzero melanoma sensitivity: 99.19% (123/124) at (E=0.333, C=0.250, S=0.417), macro AUC 0.98103, acc 87.96%.
- ES (the drop-one champion on sensitivity): 99.19% (123/124), macro AUC 0.97722, acc 85.84%.

## 4. Robustness -- contaminated 727 basis (all cached rows, no manifest filter)

n=727, melanoma n=188

| config | mel TP | mel sensitivity | Wilson 95% CI | accuracy |
|---|---|---|---|---|
| E | 185/188 | 98.40% | [95.41%, 99.46%] | 84.73% |
| C | 184/188 | 97.87% | [94.66%, 99.17%] | 89.13% |
| S | 185/188 | 98.40% | [95.41%, 99.46%] | 85.97% |
| EC | 184/188 | 97.87% | [94.66%, 99.17%] | 88.45% |
| ES | 186/188 | 98.94% | [96.20%, 99.71%] | 86.24% |
| CS | 186/188 | 98.94% | [96.20%, 99.71%] | 89.13% |
| ECS | 185/188 | 98.40% | [95.41%, 99.46%] | 88.31% |

McNemar ECS vs ES on melanoma, contaminated basis: b=1, c=2, exact two-sided p = 1.0000

## 5. Replication on the validation split

`cache_*_val.npz` exists: n=748, melanoma n=202. No clean manifest covers val, so this is the full (unfiltered) validation split -- the analogue of the contaminated basis, not the clean 565.

| config | mel TP | mel sensitivity | Wilson 95% CI | accuracy | macro AUC |
|---|---|---|---|---|---|
| E | 200/202 | 99.01% | [96.46%, 99.73%] | 84.22% | 0.96853 |
| C | 199/202 | 98.51% | [95.73%, 99.49%] | 88.64% | 0.96257 |
| S | 202/202 | 100.00% | [98.13%, 100.00%] | 84.49% | 0.96960 |
| EC | 199/202 | 98.51% | [95.73%, 99.49%] | 87.97% | 0.97692 |
| ES | 202/202 | 100.00% | [98.13%, 100.00%] | 85.03% | 0.97371 |
| CS | 200/202 | 99.01% | [96.46%, 99.73%] | 88.37% | 0.97690 |
| ECS | 199/202 | 98.51% | [95.73%, 99.49%] | 87.03% | 0.97822 |

McNemar ECS vs ES on validation melanoma: b=0 (ECS right, ES wrong), c=3 (ECS wrong, ES right), exact two-sided p = 0.2500

Validation on all 748 images:
  b=26, c=11, exact two-sided p = 0.0201

### Direction of the ES-over-ECS melanoma advantage across all three bases

| basis | n melanoma | ECS sens | ES sens | ES - ECS | McNemar b / c | exact p |
|---|---|---|---|---|---|---|
| clean test 565 | 124 | 97.58% | 99.19% | +1.61 pp | 0 / 2 | 0.5000 |
| contaminated test 727 | 188 | 98.40% | 98.94% | +0.53 pp | 1 / 2 | 1.0000 |
| validation 748 | 202 | 98.51% | 100.00% | +1.49 pp | 0 / 3 | 0.2500 |

## 6. Pooled evidence and the identity of the misses

### Pooled melanoma cases, clean test 565 + full validation 748

These are disjoint image sets, so the discordant pairs pool additively.
- n melanoma pooled = 124 + 202 = 326
- b (ECS right, ES wrong) = 0;  c (ECS wrong, ES right) = 5;  b+c = 5
- **Exact two-sided binomial p = 0.0625**  (one-sided favouring ES: 0.0312)
- Pooled ECS sensitivity 98.16% (320/326), ES 99.69% (325/326), difference +1.53 pp
- Even pooled over 326 melanoma cases the test does not reach p<0.05: every discordant pair falls the same way, but there are only 5 of them, and 5 unanimous discordances give an exact p of 0.0625 at best.

### How many consistent discordant pairs would be needed

| b+c (all favouring ES) | exact two-sided p |
|---|---|
| 0 / 2 | 0.5000 |
| 0 / 3 | 0.2500 |
| 0 / 4 | 0.1250 |
| 0 / 5 | 0.0625 |
| 0 / 6 | 0.0312 |
| 0 / 7 | 0.0156 |
| 0 / 8 | 0.0078 |
| 0 / 9 | 0.0039 |
| 0 / 10 | 0.0020 |

Significance at alpha=0.05 needs at least 6 discordant melanoma cases all falling the same way. We have 5 across 326 melanoma images, i.e. the current sample size cannot resolve a difference this small no matter how consistent it is.

### The individual melanoma misses (clean 565)

| image | true | ECS pred | ECS p(mel) | ES pred | ES p(mel) | E p(mel) | C p(mel) | S p(mel) |
|---|---|---|---|---|---|---|---|---|
| ISIC_6966105.jpg | Melanoma | Basal_Cell_Carcinoma | 0.3427 | Melanoma | 0.4922 | 0.4397 | 0.0436 | 0.5447 |
| ISIC_7406359.jpg | Melanoma | Keratosis | 0.0645 | Keratosis | 0.0843 | 0.0458 | 0.0248 | 0.1229 |
| ISIC_7619041.jpg | Melanoma | Keratosis | 0.3690 | Melanoma | 0.5111 | 0.1688 | 0.0846 | 0.8535 |

The entire ECS-vs-ES melanoma gap is 3 images. ConvNeXt's own melanoma probability on the cases ECS loses is what drags the three-way average below the decision boundary -- note that ConvNeXt alone (C) misses 3 melanomas too, the same count as ECS.

### What switching ECS -> ES would cost elsewhere

- clean test 565: accuracy 87.79% (ECS) vs 85.84% (ES), -1.95 pp; McNemar b=18, c=7, exact two-sided p = 0.0433
  per-class recall change (ES - ECS):
    Basal_Cell_Carcinoma    +0.00 pp   (n=106)
    Eczema                 -10.81 pp   (n=111)
    Keratosis               -0.75 pp   (n=133)
    Melanoma                +1.61 pp   (n=124)
    Tinea                   +0.00 pp   (n=91)
- validation 748: accuracy 87.03% (ECS) vs 85.03% (ES), -2.01 pp; McNemar b=26, c=11, exact two-sided p = 0.0201
  per-class recall change (ES - ECS):
    Basal_Cell_Carcinoma    -3.36 pp   (n=119)
    Eczema                  -2.70 pp   (n=148)
    Keratosis               -4.40 pp   (n=159)
    Melanoma                +1.49 pp   (n=202)
    Tinea                   -2.50 pp   (n=120)

### Macro AUC, the threshold-free comparison

- clean test 565: macro AUC ECS 0.98132 vs ES 0.97722 (-0.00411); melanoma-vs-rest AUC ECS 0.99872 vs ES 0.99664 (-0.00208)
- validation 748: macro AUC ECS 0.97822 vs ES 0.97371 (-0.00451); melanoma-vs-rest AUC ECS 0.99679 vs ES 0.99655 (-0.00024)

ECS wins macro AUC on both bases. Macro AUC is threshold-free and uses all 565/748 images rather than the 2-3 discordant melanoma cases, so it is the far better-powered statistic here -- and it points the opposite way to the raw melanoma-sensitivity ranking.

### A cheaper lever than reweighting: the melanoma decision threshold

Every melanoma ECS misses is a case where p(melanoma) is high but not the argmax. Escalating on p(melanoma) >= t instead of requiring argmax recovers them without touching the weights.

| t | ECS mel sensitivity | ECS mel TP | false-positive melanoma flags (non-mel images) | overall acc impact basis |
|---|---|---|---|---|
| 0.50 | 97.58% | 121/124 | 2/441 (0.5%) | n=565 |
| 0.40 | 97.58% | 121/124 | 3/441 (0.7%) | n=565 |
| 0.35 | 98.39% | 122/124 | 3/441 (0.7%) | n=565 |
| 0.30 | 99.19% | 123/124 | 4/441 (0.9%) | n=565 |
| 0.25 | 99.19% | 123/124 | 7/441 (1.6%) | n=565 |
| 0.20 | 99.19% | 123/124 | 7/441 (1.6%) | n=565 |
| 0.15 | 99.19% | 123/124 | 8/441 (1.8%) | n=565 |
| 0.10 | 99.19% | 123/124 | 16/441 (3.6%) | n=565 |
