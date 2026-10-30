# new_results.md — measured results from the recovery retrains

Every number in this file was produced by a notebook in `paper/experiments/` and
independently re-derived from the raw confusion matrix before being written down.
**This file is the source of truth for the Results section of `paper/main.tex`.**
Where a number here disagrees with `main.tex`, `main.tex` is wrong.

Companions: `FACT_CHECK.md` (the audit), `REVISION_NOTES.md` (the rewrite plan),
`CHANGELOG.md` (what changed under `paper/`).

Last updated 2026-10-29.

---

## Status

| Notebook | Backbone | Status | Artefacts received |
|---|---|---|---|
| `05_efficientnet_retrain.ipynb` | EfficientNet-B1 | **complete, verified** | `results.zip` (9 files), 2026-10-29 |
| `02_dedup_phash.ipynb` | — | **complete** | 2026-10-29 — **162/727 test images (22.28%) leak into train/val**; clean manifest = 565 |
| `06_swin_retrain.ipynb` | Swin-Tiny | **complete, verified** | `results (1).zip` (9 files), 2026-10-29 |
| `04_convnext_retrain.ipynb` | ConvNeXt-Tiny | **complete, verified** — cell 17 did not run (see §1c.9) | console output, 2026-10-30 |
| `01_eval_and_ablation.ipynb` | ensemble | **ready to run** — patched to resolve `*_v2.pth`; needs the three checkpoints uploaded (see Source files) | — |
| `03_skin_tone_ita.ipynb` | — | blocked on NB1 caches | — |

Nothing in this file speaks to the **ensemble**. Every ensemble number in the paper
(accuracy, macro-AUC, MCC, the ablation table, the paired bootstrap) remains
unmeasured until NB1 has run. **All three retrains are now done**; what remains is to
collect the three v2 checkpoints plus NB2's clean manifest in one place and run NB1.

NB1 would have failed at cell 4 before this was noticed: its `_resolve()` calls named
only the **original** filenames (`efficientnet_b1_5cls_skin.pth`), while the retrains
wrote `*_v2.pth`. That is the same failure that stopped NB4's cell 17 — a missing
candidate path, raised before any work is done. Cell 4 now tries the v2 name first
and refuses outright if the three resolved paths are not all the same generation,
because an ensemble built from one original and two retrains would be meaningless
and would otherwise run silently. The rest of NB1 needed no change: `load_backbone()`
already reads `dropout` and `drop_path_rate` from `ckpt['config']`, so the v2 recipes
load under `strict=True`, and the §6 reproduction gate compares against each
checkpoint's own stored `test_acc` / `macro_auc`, so it re-derives the v2 values
rather than the paper's.

### The test set this file reports on

**Every metric in §1 and §1b is computed on the full 727-image test set**, which is
the basis NB5 and NB6 evaluate against. NB2 has since shown that **22.28% of those 727
images have a near-duplicate in train or val**, so these figures are measured on a
contaminated set and are expected to fall when re-scored on the clean 565.

This is not a defect in either retrain — their job is to produce checkpoints that
verify, and both did. But it means **the numbers in §1 and §1b are not the ones that
go in the paper.** The reportable figures come from `01_eval_and_ablation.ipynb`,
which now applies `test_clean_manifest.csv` as its primary basis. Expect the clean
accuracies to be **lower** than 84.73% / 85.97%; the gap is the measurement bias the
screening removes, and §15 of NB1 quantifies it.

Treat §1 and §1b as: *the retrains reproduce the recipe and the checkpoints are
sound.* Treat NB1's forthcoming output as: *the performance the paper claims.*

The one exception is NB1's **reproduction gate**, which deliberately runs on the full
727 because that is the basis each checkpoint stored its own accuracy on.

---

## 1. EfficientNet-B1 — recovery retrain

### 1.1 Provenance

| Field | Value |
|---|---|
| Notebook | `05_efficientnet_retrain.ipynb` |
| Reason | recovery retrain; the shipped checkpoint did not reproduce its stored metrics |
| Recipe | verbatim from `src/models/efficientnet_b1_5class.ipynb` |
| Seed | 42 |
| torch / timm | 2.11.0+cpu / 1.0.29 |
| Pretrained weights | `timm/efficientnet_b1.ra4_e3600_r240_in1k` @ `4e4a9c0` |
| Data root | `IPD_Datasets_Final_Split` (Kaggle) |
| Generated (UTC) | checkpoint `provenance.generated_utc` 2026-10-29T08:16:54Z; `results_efficientnet_v2.json` `meta.generated_utc` 2026-10-29T08:17:18Z (written 24 s later, after test evaluation) |
| Checkpoint | `efficientnet_b1_5cls_skin_v2.pth` (27.83 MB, 512 tensors) |

### 1.2 Headline test metrics — n = 727 (contaminated basis; see "The test set this file reports on")

| Metric | Value |
|---|---|
| Accuracy | **84.73%** (616/727), Wilson 95% CI [81.94, 87.16] |
| Macro AUC | **0.9704** |
| MCC | **0.8113** |
| Balanced accuracy | **0.8469** |
| Macro F1 | **0.8376** |

The shipped checkpoint stored 85.11% / 0.9636 but did not reproduce them (it scored
66.99% — see `REVISION_NOTES.md` §0). The retrain lands **0.38 pp below** the
originally claimed accuracy and **0.0068 above** the claimed macro-AUC. Both of the
retrain's values are reproducible from the saved weights; neither of the original's
was.

### 1.3 Per-class test metrics

| Class | Support | Recall (sens.) | Recall 95% CI | Precision | Specificity | AUC |
|---|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | 121 | **0.9917** | [0.9547, 0.9985] | 0.8696 | 0.9703 | 0.9949 |
| Eczema | 145 | **0.8276** | [0.7579, 0.8804] | 0.7500 | 0.9313 | 0.9591 |
| Keratosis | 158 | **0.6139** | [0.5362, 0.6863] | 0.9065 | 0.9824 | 0.9397 |
| Melanoma | 188 | **0.9840** | [0.9541, 0.9946] | 0.9536 | 0.9833 | 0.9943 |
| Tinea | 115 | **0.8174** | [0.7369, 0.8773] | 0.7344 | 0.9444 | 0.9640 |

CIs are Wilson score intervals computed here, not by the notebook.

### 1.4 Clinical sensitivity targets (`src/models/CLAUDE.md`)

| Class | Measured | Target | Met |
|---|---|---|---|
| Melanoma | 98.40% | ≥ 95% | ✅ |
| Basal_Cell_Carcinoma | 99.17% | ≥ 90% | ✅ |
| Eczema | 82.76% | ≥ 80% | ✅ |
| Tinea | 81.74% | ≥ 80% | ✅ |
| **Keratosis** | **61.39%** | ≥ 80% | ❌ **−18.6 pp** |

NPV, target ≥ 0.95: **Melanoma 0.9944**, **BCC 0.9983**. Both met.

### 1.5 Confusion matrix — counts (rows = true, columns = predicted)

| True ↓ / Pred → | BCC | Eczema | Keratosis | Melanoma | Tinea |
|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | **120** | 0 | 1 | 0 | 0 |
| Eczema | 0 | **120** | 2 | 0 | 23 |
| Keratosis | 18 | 23 | **97** | 9 | 11 |
| Melanoma | 0 | 0 | 3 | **185** | 0 |
| Tinea | 0 | 17 | 4 | 0 | **94** |

### 1.6 Confusion matrix — row-normalised (%)

| True ↓ / Pred → | BCC | Eczema | Keratosis | Melanoma | Tinea |
|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | **99.2** | 0.0 | 0.8 | 0.0 | 0.0 |
| Eczema | 0.0 | **82.8** | 1.4 | 0.0 | 15.9 |
| Keratosis | 11.4 | 14.6 | **61.4** | 5.7 | 7.0 |
| Melanoma | 0.0 | 0.0 | 1.6 | **98.4** | 0.0 |
| Tinea | 0.0 | 14.8 | 3.5 | 0.0 | **81.7** |

### 1.7 Paper Sec. 6.3 confusion claims — corrected

These are the single-backbone EfficientNet figures. The paper's claims were made
about the ensemble, so they are not strictly comparable until NB1 runs, but the
direction of each error is already measurable and two of the three are overstated.

| Paper claim | Measured (EfficientNet-B1) |
|---|---|
| Eczema → Tinea 22% | **15.9%** (23/145) |
| Tinea → Eczema 19% | **14.8%** (17/115) |
| Keratosis → BCC 10% | **11.4%** (18/158) |

### 1.8 Malignant / benign leakage

The clinically asymmetric view of the same matrix:

| Direction | Rate |
|---|---|
| True malignant (Melanoma, BCC) predicted benign | **1.29%** (4/309) |
| True benign predicted malignant | **6.46%** (27/418) |

The model errs ~5× more often toward over-referral than under-referral. That is the
intended direction for a screening tool and is worth stating explicitly — all four
missed malignancies are Melanoma→Keratosis (3) and BCC→Keratosis (1).

### 1.9 Training dynamics

| Field | Value |
|---|---|
| Epochs run | **25 of 30 configured** (early stop) |
| Saved epoch | **19** |
| Best composite | **0.9310663419** |
| Val acc @ saved epoch | 84.22% |
| Melanoma sens. @ saved epoch | 99.01% |
| BCC sens. @ saved epoch | 94.12% |
| Train−val gap, last 5 epochs | +1.34% to +4.02%, mean **2.62%** (target < 5%) |

Composite criterion `0.4·mel_sens + 0.3·bcc_sens + 0.3·val_acc` re-derived from the
saved epoch's components: **0.9310663419** — exact match to the stored
`best_composite`. Early stop is arithmetically correct: last improvement at epoch 19,
`patience = 6`, so the run ends at epoch 25. Epochs 20–25 all scored below the
epoch-19 composite.

### 1.10 Splits and class balance

| Split | Images |
|---|---|
| Train | 5,180 |
| Val | 748 |
| Test | 727 |
| **Total** | **6,655** |

Actual ratio ≈ **78 / 11 / 11**, inherited as a fixed folder tree — **not** the
70/15/15 the paper claims, and the corpus is 6,655 images across 5 classes, **not**
14,498.

Train distribution: BCC 968 · Eczema 1,072 · Keratosis 977 · Melanoma 1,197 · Tinea 966.

Loss weights (`1/√count`, then clinical boosts ×2 Mel, ×2 BCC, ×3 Tinea, ×1.5 Eczema,
renormalised) — re-derived from the train distribution and matching the stored values
to 0.0 absolute error:

| Class | Weight |
|---|---|
| Basal_Cell_Carcinoma | 0.21686301 |
| Eczema | 0.15455641 |
| Keratosis | 0.10793091 |
| Melanoma | 0.19501862 |
| Tinea | 0.32563105 |

Note Keratosis receives the **lowest** loss weight (no clinical boost, and it is not a
small class) — relevant to §3 below.

### 1.11 Measured hardware and wall-clock

This replaces the paper's unsupported "Kaggle 2×T4, 32 GB, ≈6 h".

| Field | Value |
|---|---|
| Device | **CPU** (`torch 2.11.0+cpu`, `vram_gb: null`) |
| Wall-clock | **3.78 h** (13,601.5 s) for 25 epochs |
| Mean epoch | 9.07 min |
| Frozen phase (ep 1–6) | 4.55 min/epoch ≈ 843 ms/step |
| Fine-tune phase (ep 7–25) | 10.49 min/epoch ≈ 1,943 ms/step |
| Batch size | 16 (324 steps/epoch) |
| Mixed precision | **not active** — see §2.2 |

Timing is internally consistent: summing `epoch_seconds` from `history.csv` gives
13,601.35 s against a reported 13,601.50 s wall clock (0.15 s of non-epoch overhead).
The 2.3× step-time jump falls exactly at epoch 7, which is where `freeze_epochs = 6`
unfreezes the backbone — the expected signature.

**Do not write "T4" for this run.** See §2.1.

### 1.12 Checkpoint integrity

The notebook's §10 save-then-verify cell passed:

| Check | Result |
|---|---|
| Tensors compared | 512, all bit-identical to the evaluated model |
| `test_acc` in memory | 0.8473177441540578 |
| `test_acc` reloaded from disk | 0.8473177441540578 |
| `test_acc` stored in checkpoint | 0.8473177441540578 |

Independently confirmed by parsing both `.pth` files directly: 512 tensor storages,
`_metadata` present (so `load_state_dict(strict=True)` succeeds), `classes` is exactly
`['Basal_Cell_Carcinoma','Eczema','Keratosis','Melanoma','Tinea']` in the order
`src/models/ensemble.py` requires, and the `test_acc` / `macro_auc` floats embedded in
the pickle match this file to full precision.

**The original checkpoints would have failed check 1.** This is the failure mode the
retrains exist to fix.

### 1.13 `best_val_acc` convention — fixed vs the originals

The original notebooks ended with `best_val_acc = max(history['val_acc'])`, i.e. the
best validation accuracy over **all** epochs, which need not be the epoch whose
weights were saved. Verified by loading the shipped checkpoints:

| Checkpoint | `saved_epoch` | stored `best_val_acc` | `max(history.val_acc)` | at epoch |
|---|---|---|---|---|
| `efficientnet_b1_5cls_skin.pth` (orig) | absent | 0.8768939394 | 0.8768939394 | 27 |
| `convnext_tiny_5cls_skin.pth` (orig) | absent | 0.7556818182 | 0.7556818182 | 8 |
| `swin_tiny_5cls_skin.pth` (orig) | absent | 0.8835227273 | 0.8835227273 | 27 |
| `efficientnet_b1_5cls_skin_v2.pth` (NB5) | **19** | **0.8422459893** | 0.8768939394* | — |

\* the v2 run has its own history; the point is that v2's `best_val_acc` equals the
**saved epoch's** val accuracy (0.8422, §1.9), not the maximum over all epochs.

This matters because `src/models/CLAUDE.md` documents `best_val_acc` as "val_acc of
the best-composite epoch" — which the originals did **not** honour and NB5 now does.
The originals' value is therefore not a property of the shipped weights. The ConvNeXt
case is the clearest: composite selected epoch 4, but `best_val_acc` came from epoch 8.

### 1.14 Independent re-derivation of every number in §1

Nothing in §1 is taken from the notebook's own summary fields. Each value was
recomputed from the raw 5×5 confusion matrix, the train distribution and
`history.csv`, then compared against what this file claims. All agree:

| Group | Checks | Result |
|---|---|---|
| `confusion_test.csv` ≡ JSON matrix | 1 | exact |
| accuracy / balanced acc / MCC (Gorodkin) / macro-F1 from CM | 5 | ≤ 2e-16 |
| per-class recall / precision / specificity from CM | 20 | exact |
| `per_class_test.csv` ≡ JSON | 25 | exact |
| row-normalised CM (all 25 cells) | 25 | exact to 1 dp |
| §1.7 confusion claims, §1.8 leakage rates | 14 | exact |
| composite = 0.4·mel + 0.3·bcc + 0.3·val_acc at saved epoch | 3 | 0.9310663419283105 |
| early stop arithmetic (argmax = ep 19, 19 + patience 6 = 25) | 4 | exact |
| train−val gap range and mean, last 5 epochs | 3 | exact |
| loss weights re-derived from `1/√count` + boosts | 6 | ≤ 1e-7 |
| timing: Σ`epoch_seconds` vs wall-clock, phase means, ms/step | 11 | ≤ 0.02 s |
| Wilson 95% CIs (accuracy + all 5 recalls) | 12 | ≤ 1e-4 |
| NPV Melanoma / BCC | 2 | exact |
| checkpoint: 512 tensors, `_metadata`, class order, 0 fusion tensors | 4 | pass |
| bare vs wrapped `.pth` — all 512 tensors | 1 | bit-identical |

Two apparent mismatches were run down and are **not** errors:

- **File sizes.** 27.83 MB / 27.82 MB are decimal MB (10⁶ bytes); the same files are
  26.54 / 26.53 MiB. Both correct under their own convention.
- **Timestamps.** `08:16:54Z` is the checkpoint's `provenance.generated_utc`;
  `08:17:18Z` is the results JSON's `meta.generated_utc`, written after evaluation.

---

## 1b. Swin-Tiny — recovery retrain

### 1b.1 Provenance

| Field | Value |
|---|---|
| Notebook | `06_swin_retrain.ipynb` |
| Reason | recovery retrain; the shipped checkpoint did not reproduce its stored metrics |
| Recipe | verbatim from `src/models/swin_tiny_5class.ipynb` |
| Seed | 42 |
| torch / timm | **2.11.0+cu128** / 1.0.29 |
| Pretrained weights | `timm/swin_tiny_patch4_window7_224.ms_in1k` @ `4cc3a72` |
| Device | **Tesla T4**, 14.56 GB VRAM |
| Generated (UTC) | 2026-10-29T10:24:02Z |
| Checkpoint | `swin_tiny_5cls_skin_v2.pth` (111.0 MB, 177 tensors) |

### 1b.2 Headline test metrics — n = 727 (contaminated basis)

| Metric | Value |
|---|---|
| Accuracy | **85.97%** (625/727), Wilson 95% CI [83.26, 88.31] |
| Macro AUC | **0.9737** |
| MCC | **0.8252** |
| Balanced accuracy | **0.8567** |
| Macro F1 | **0.8505** |

The shipped checkpoint stored 85.39% / 0.9683 but did not reproduce them (it scored
65.75% — `REVISION_NOTES.md` §0). The retrain lands **0.58 pp above** the originally
claimed accuracy and **0.0054 above** the claimed macro-AUC, and unlike the original
both values are reproducible from the saved weights.

### 1b.3 Per-class test metrics

| Class | Support | Recall (sens.) | Recall 95% CI | Precision | Specificity | AUC |
|---|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | 121 | **0.9835** | [0.9417, 0.9955] | 0.8440 | 0.9637 | 0.9953 |
| Eczema | 145 | **0.8414** | [0.7732, 0.8919] | 0.7722 | 0.9381 | 0.9679 |
| Keratosis | 158 | **0.6835** | [0.6074, 0.7510] | 0.8710 | 0.9719 | 0.9436 |
| Melanoma | 188 | **0.9840** | [0.9541, 0.9946] | 0.9788 | 0.9926 | 0.9985 |
| Tinea | 115 | **0.7913** | [0.7082, 0.8556] | 0.7913 | 0.9608 | 0.9634 |

### 1b.4 Clinical sensitivity targets

| Class | Measured | Target | Met |
|---|---|---|---|
| Melanoma | 98.40% | ≥ 95% | ✅ |
| Basal_Cell_Carcinoma | 98.35% | ≥ 90% | ✅ |
| Eczema | 84.14% | ≥ 80% | ✅ |
| **Tinea** | **79.13%** | ≥ 80% | ❌ **−0.87 pp** |
| **Keratosis** | **68.35%** | ≥ 80% | ❌ **−11.6 pp** |

NPV: **Melanoma 0.9944**, **BCC 0.9966**. Both ≥ 0.95.

Tinea misses its target by less than one point, and its CI [70.82, 85.56] straddles
80% — do not report it as a clear failure, report it as at-target within measurement
error.

### 1b.5 Confusion matrix — counts (rows = true, columns = predicted)

| True ↓ / Pred → | BCC | Eczema | Keratosis | Melanoma | Tinea |
|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | **119** | 0 | 1 | 1 | 0 |
| Eczema | 0 | **122** | 7 | 0 | 16 |
| Keratosis | 22 | 17 | **108** | 3 | 8 |
| Melanoma | 0 | 0 | 3 | **185** | 0 |
| Tinea | 0 | 19 | 5 | 0 | **91** |

### 1b.6 Confusion matrix — row-normalised (%)

| True ↓ / Pred → | BCC | Eczema | Keratosis | Melanoma | Tinea |
|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | **98.3** | 0.0 | 0.8 | 0.8 | 0.0 |
| Eczema | 0.0 | **84.1** | 4.8 | 0.0 | 11.0 |
| Keratosis | 13.9 | 10.8 | **68.4** | 1.9 | 5.1 |
| Melanoma | 0.0 | 0.0 | 1.6 | **98.4** | 0.0 |
| Tinea | 0.0 | 16.5 | 4.3 | 0.0 | **79.1** |

### 1b.7 Paper Sec. 6.3 confusion claims

| Paper claim | Swin-Tiny | EfficientNet-B1 |
|---|---|---|
| Eczema → Tinea 22% | **11.0%** (16/145) | 15.9% |
| Tinea → Eczema 19% | **16.5%** (19/115) | 14.8% |
| Keratosis → BCC 10% | **13.9%** (22/158) | 11.4% |

Neither backbone supports the 22% figure. Both exceed the 10% Keratosis→BCC figure.

### 1b.8 Malignant / benign leakage

| Direction | Rate |
|---|---|
| True malignant predicted benign | **1.29%** (4/309) |
| True benign predicted malignant | **5.98%** (25/418) |

Identical malignant-miss rate to EfficientNet (4/309), but a different composition:
3 Melanoma→Keratosis plus **1 BCC→Melanoma** (an error that stays within the
malignant set and would still be referred) and 1 BCC→Keratosis.

### 1b.9 Training dynamics — the saved epoch is the last epoch

| Field | Value |
|---|---|
| Epochs run | **30 of 30** — ran to completion, no early stop |
| Saved epoch | **30 (the final epoch)** |
| Best composite | **0.9433919022** |
| Val acc @ saved epoch | 84.49% |
| Melanoma sens. @ saved epoch | **100.00%** |
| BCC sens. @ saved epoch | 96.64% |
| Train−val gap, last 5 epochs | +1.46% to +3.69%, mean **2.70%** (target < 5%) |

**This is the one result that needs a caveat.** The composite criterion peaked at the
very last epoch, so the run was still improving when the 30-epoch budget ran out.
Early stopping never fired (`patience = 5`; the longest run of non-improving epochs
was 4, at epochs 24–25 and 27–29). Best val accuracy came at epoch 27 (84.89%), and
the composite trajectory over the last eight epochs — 0.9398, 0.9322, 0.9389, 0.9410,
0.9401, 0.9381, 0.9398, **0.9434** — is flat-with-noise, not climbing.

So the model is **converged but not demonstrably at its optimum**. It is not
overfitting: the final gap is 3.69%, inside the 5% target, and val loss is still
drifting down. A longer budget would probably buy a little more. Two honest options:

1. **Report as-is** and state that Swin reached its configured 30-epoch budget with
   the composite still marginally improving. Defensible, costs nothing, and keeps the
   recipe byte-identical to the original — which is the stated design rule.
2. **Re-run with more epochs.** Breaks the byte-identical-recipe rule, costs another
   T4 hour, and on this evidence would gain well under a point.

**Recommendation: option 1.** The flat trajectory says there is little left on the
table, and recipe fidelity is worth more to the paper than a fractional gain. But the
sentence must appear in Methods — a reviewer who reads `saved_epoch == num_epochs`
will otherwise ask.

### 1b.10 Measured hardware and wall-clock

| Field | Value |
|---|---|
| Device | **Tesla T4** (cuda), 14.56 GB |
| Wall-clock | **0.588 h** (2,116.0 s) for 30 epochs |
| Mean epoch | **1.18 min** |
| Frozen phase (ep 1–8) | 67.6 s/epoch ≈ 104 ms/step |
| Fine-tune phase (ep 9–30) | 71.6 s/epoch ≈ 110 ms/step |
| Batch size | 8 (648 steps/epoch) |
| Mixed precision | **active** (fp16 autocast + GradScaler) |

Σ`epoch_seconds` = 2,115.87 s vs 2,116.02 s reported (0.15 s overhead) — consistent.

**This ran ~6× faster than `paper/experiments/README.md` estimates** (3.5–4.5 h). The
README figure was an unmeasured guess; 0.59 h is the measured value. Note the frozen
→ unfrozen step-time jump is only 1.06× here, versus 2.3× for EfficientNet on CPU —
on a T4 at batch 8 the run is data-loader-bound, not compute-bound, which is also why
Swin finished faster in wall-clock than the *smaller* EfficientNet did on CPU.

**Revise the README's GPU-hour budget.** Swin cost 0.59 GPU-h against a 4 h
allocation. The ~9 GPU-h/week Kaggle budget is nowhere near binding.

### 1b.11 Checkpoint integrity

§10 save-then-verify passed:

| Check | Result |
|---|---|
| Tensors compared | **177**, all bit-identical to the evaluated model |
| `test_acc` in memory | 0.859697386519945 |
| `test_acc` reloaded from disk | 0.859697386519945 |
| `test_acc` stored in checkpoint | 0.859697386519945 |

Independently confirmed by parsing the `.pth` directly: 177 tensor storages,
`_metadata` present (`strict=True` will load), `classes` exactly
`['Basal_Cell_Carcinoma','Eczema','Keratosis','Melanoma','Tinea']`, no fusion
tensors. Tensors are stored on `cuda:0` — harmless, since every loader passes
`map_location`.

### 1b.12 Swin vs EfficientNet — same 727 images, same splits, same seed

| Metric | EfficientNet-B1 | Swin-Tiny | Δ |
|---|---|---|---|
| Accuracy | 0.8473 | **0.8597** | **+0.0124** |
| Macro AUC | 0.9704 | **0.9737** | +0.0033 |
| MCC | 0.8113 | **0.8252** | +0.0139 |
| Balanced accuracy | 0.8469 | **0.8567** | +0.0098 |
| Macro F1 | 0.8376 | **0.8505** | +0.0129 |

Per-class recall:

| Class | EfficientNet-B1 | Swin-Tiny | Δ |
|---|---|---|---|
| Basal_Cell_Carcinoma | **0.9917** | 0.9835 | −0.0083 |
| Eczema | 0.8276 | **0.8414** | +0.0138 |
| **Keratosis** | 0.6139 | **0.6835** | **+0.0696** |
| Melanoma | 0.9840 | 0.9840 | 0.0000 |
| Tinea | **0.8174** | 0.7913 | −0.0261 |

Swin wins on every aggregate metric, and the bulk of the margin is **Keratosis
recall, +7.0 pp**. The two models are *identical* on Melanoma (185/188 each — and on
the same failure mode, Melanoma→Keratosis) and trade places on BCC and Tinea. That
complementarity is exactly the argument for the ensemble, and it is now measurable
rather than asserted.

**Do not call this a significant difference yet.** +1.24 pp on n=727 with CIs that
overlap heavily ([81.94, 87.16] vs [83.26, 88.31]) is not a significance claim. NB1's
paired bootstrap is the test that settles it, and it runs on the clean 565.

**This section compares two of three backbones only.** ConvNeXt v2 was retrained after
it was written and beats both (§1c.5), so "Swin wins on every aggregate" holds within
this pair and not overall. The complementarity point above also needs rereading in
that light: see the §4 row withdrawing the paper's version of it.

### 1b.13 Independent re-derivation of every number in §1b

As with §1.14, nothing in §1b is taken from the notebook's own summary fields. Every
value was recomputed from the raw 5×5 confusion matrix, the train distribution and
`history.csv`, then compared against what this file claims:

| Group | Checks | Result |
|---|---|---|
| `confusion_test.csv` ≡ JSON matrix; row sums ≡ supports | 6 | exact |
| accuracy / balanced acc / MCC (Gorodkin) / macro-F1 from CM | 5 | ≤ 2e-16 |
| per-class recall / precision / specificity from CM | 15 | exact |
| `per_class_test.csv` ≡ JSON | 25 | exact |
| row-normalised CM (all 25 cells) | 25 | exact to 1 dp |
| §1b.7 confusion claims, §1b.8 leakage rates, NPVs | 10 | exact |
| composite = 0.4·mel + 0.3·bcc + 0.3·val_acc at saved epoch | 3 | 0.9433919022154316 |
| saved_epoch = argmax(composite) = 30; no early stop (max non-improving run = 4 < patience 5) | 4 | exact |
| train−val gap range and mean, last 5 epochs | 3 | exact |
| loss weights re-derived from `1/√count` + boosts | 5 | 0.0 |
| timing: Σ`epoch_seconds` vs wall-clock, phase means, ms/step | 8 | ≤ 0.15 s |
| Wilson 95% CIs (accuracy + all 5 recalls) | 12 | ≤ 1e-4 |
| splits sum to 6,655; identical to NB5's splits | 4 | exact |
| checkpoint: 177 tensors, `_metadata`, class order, 0 fusion tensors | 4 | pass |
| §1b.12 cross-model deltas | 10 | exact |

The loss weights, splits and train distribution are **identical** to NB5's, which is
the expected result — both notebooks read the same fixed folder tree and construct
weights the same way. That agreement is itself a check that neither run silently
re-split the data.

---

## 1c. ConvNeXt-Tiny — recovery retrain

### 1c.1 Provenance

Console output pasted by the author, 2026-10-30, from `04_convnext_retrain.ipynb` run
on Kaggle after the patch described in `CHANGELOG.md`. **The archive has not been
received**, so unlike §1 and §1b the numbers below were not re-derived from a raw
confusion matrix — they are re-derived from the reported per-class table (§1c.10),
which is a weaker check. Treat §1c as verified-arithmetic, not verified-artefact,
until the zip lands.

The config changes from the original run, carried by the patch: `freeze_epochs`
8 → 3, `patience` 5 → 10, `num_epochs` 30 → 40.

### 1c.2 Headline test metrics — n = 727 (contaminated basis)

| Metric | ConvNeXt v2 | Original | Δ |
|---|---|---|---|
| Test accuracy | **89.13%** (648/727) | 71.35% | **+17.78 pp** |
| Macro AUC | **0.9684** | 0.9418 | +0.0266 |
| MCC | **0.8650** | — (paper claimed 0.6507) | +0.2143 vs claim |
| Balanced accuracy | **89.04%** | — | — |
| Macro F1 | **0.8847** | — | — |

**This is the largest single correction in the paper.** The original ConvNeXt was not
architecturally weak — it received roughly one epoch of fine-tuning (§2.3). Given a
fair schedule it becomes the **best of the three backbones**, beating Swin by 3.17 pp
of accuracy.

### 1c.3 Per-class test metrics

| Class | Support | Recall | Precision | F1 | Specificity | AUC |
|---|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | 121 | **1.0000** | 0.8582 | 0.9237 | 0.9670 | 0.9893 |
| Eczema | 145 | 0.8828 | 0.8101 | 0.8449 | 0.9485 | 0.9613 |
| Keratosis | 158 | 0.7468 | 0.9365 | 0.8310 | 0.9859 | 0.9348 |
| Melanoma | 188 | 0.9787 | 0.9892 | 0.9840 | 0.9963 | 0.9988 |
| Tinea | 115 | 0.8435 | 0.8362 | 0.8398 | 0.9690 | 0.9578 |

### 1c.4 Clinical sensitivity targets (`src/models/CLAUDE.md`)

| Class | Target | Measured | Verdict |
|---|---|---|---|
| Melanoma | ≥ 95% | **97.87%** | **pass** |
| Basal_Cell_Carcinoma | ≥ 90% | **100.00%** | **pass** |
| Eczema | ≥ 80% | 88.28% | pass |
| Tinea | ≥ 80% | 84.35% | pass |
| Keratosis | ≥ 80% | 74.68% | **fail** (−5.3 pp) |

Four of five targets met — the best of the three retrains. **BCC recall is 100%: all
121 are caught.** The cost is 20 false positives (from precision 0.8582), i.e. 20
non-BCC images routed to a BCC-positive outcome. For a screening tool that is the
right direction to err, but it should be stated rather than presented as free.

Melanoma misses 4 of 188. Keratosis remains the only failing target.

### 1c.5 ConvNeXt v2 vs the other two retrains — same 727 images, same splits, same seed

| Metric | EfficientNet-B1 | Swin-Tiny | **ConvNeXt-Tiny v2** | Best |
|---|---|---|---|---|
| Accuracy | 84.73% | 85.97% | **89.13%** | ConvNeXt |
| Macro AUC | 0.9704 | 0.9737 | **0.9684** | Swin |
| MCC | 0.8113 | 0.8252 | **0.8650** | ConvNeXt |
| Balanced accuracy | 84.69% | 85.67% | **89.04%** | ConvNeXt |
| Macro F1 | 0.8376 | 0.8505 | **0.8847** | ConvNeXt |

ConvNeXt wins four of five aggregates. **Macro AUC is the exception** — Swin ranks
better (0.9737 vs 0.9684) while classifying worse, which is the signature of a
threshold/calibration difference rather than a representation difference. NB1's
calibration and ECE work should look at this directly; it is the kind of detail that
makes an ensemble worth more than its best member.

Per-class recall, all three:

| Class | EfficientNet | Swin | ConvNeXt v2 |
|---|---|---|---|
| BCC | 0.9917 | 0.9835 | **1.0000** |
| Eczema | 0.8276 | 0.8414 | **0.8828** |
| Keratosis | 0.6139 | 0.6835 | **0.7468** |
| Melanoma | **0.9840** | **0.9840** | 0.9787 |
| Tinea | **0.8174** | 0.7913 | **0.8435** |

ConvNeXt leads on four of five classes and is the only one to reach 100% on BCC. It
is **marginally worst on Melanoma** — 184/188 versus 185/188 for the other two, a
one-image difference that no confidence interval would separate.

### 1c.6 Where the accuracy gain comes from

ConvNeXt's +3.17 pp over Swin is 23 additional correct images out of 727. By class,
versus Swin:

| Class | Swin correct | ConvNeXt correct | Δ |
|---|---|---|---|
| BCC | 119 | 121 | +2 |
| Eczema | 122 | 128 | +6 |
| Keratosis | 108 | 118 | +10 |
| Melanoma | 185 | 184 | −1 |
| Tinea | 91 | 97 | +6 |
| **total** | **625** | **648** | **+23** |

**Keratosis supplies 43% of the gain**, the same class that supplied ~70% of Swin's
margin over EfficientNet. The ranking of these three models is largely a ranking of
how well each handles Keratosis.

### 1c.7 Checkpoint integrity — verified

The save-then-verify cell added by the patch ran and passed:

| Check | Result |
|---|---|
| Tensors compared | **186, all bit-identical** |
| Accuracy in memory | 0.8913342503 |
| Accuracy after reload | 0.8913342503 |
| Accuracy stored in checkpoint | 0.8913342503 |
| `VERIFIED` | **True** |

Architecture line confirms the recipe: `features=768 | classes=5 | dropout=0.4 |
drop_path=0.2`. Written as `convnext_tiny_5cls_skin_v2.pth`, plus
`cache_convnext_tiny_v2_test.npz`.

**This closes the premise of the whole exercise.** All three shipped checkpoints
failed to reproduce their stored metrics (`REVISION_NOTES.md` §0); all three v2
checkpoints now reload bit-identically and re-score to their stored value.

### 1c.8 The three-way correction to the paper

| Backbone | Paper / stored | Measured v2 | Δ |
|---|---|---|---|
| EfficientNet-B1 | 85.11% | 84.73% | −0.38 pp |
| Swin-Tiny | 85.39% | 85.97% | +0.58 pp |
| ConvNeXt-Tiny | 71.35% | **89.13%** | **+17.78 pp** |

The first two land within ±0.6 pp of their claimed values — the original numbers were
roughly right, just not reproducible from the shipped weights. **ConvNeXt is the
outlier**, and it changes an argument rather than a number: the paper uses ConvNeXt's
weakness to argue for ensemble complementarity. That argument does not survive. See §4.

MCC 0.8650 against the paper's claimed 0.6507 is the single largest discrepancy
anywhere in this file.

### 1c.9 Cell 17 did not run — and nothing was lost

Cell 17 (the in-notebook 7-config ablation) raised `FileNotFoundError` before doing
any work. It is the only cell that reads the *other* backbones' checkpoints: it calls
`_resolve()` on `efficientnet_b1_5cls_skin_v2.pth` and `swin_tiny_5cls_skin_v2.pth`,
and `_resolve()` raises if neither candidate path exists. So without the NB5 and NB6
v2 checkpoints uploaded to `CKPT_ROOT`, the cell fails immediately — as the patch note
in the cell itself warned.

**Nothing of value was lost.** Everything through cell 15 completed: training, test
evaluation, per-class metrics, the checkpoint save, the verify, and the probability
cache. `results_convnext_v2.json` was *not* written, but it duplicates work NB1 does
properly on the clean 565-image basis. The sensible move is **not** to re-run cell 17;
it is to go straight to NB1 with all three checkpoints in place.

What is missing because of it:

- ConvNeXt's **confusion matrix** (hence the blank column in §3)
- `malignant → benign` leakage and NPV for ConvNeXt (§1.8 / §1b.8 equivalents)
- the in-notebook 7-config ablation and paired bootstrap — **NB1's job anyway**

### 1c.10 Independent re-derivation

Every aggregate recomputed from the per-class recalls and supports in §1c.3:

| Quantity | Reported | Re-derived | Agreement |
|---|---|---|---|
| Support total | 727 | 727 | exact |
| Accuracy | 0.8913342503 | 0.891332 | 2.1e-06 (4-dp rounding) |
| Correct predictions | — | 648.00 → **648/727** | exact integer |
| Balanced accuracy | 0.8904 | 0.890360 | exact to 4 dp |
| Macro F1 | 0.8847 | 0.884659 | exact to 4 dp |
| Macro AUC | 0.9684 | 0.968400 | exact |
| Macro precision | 0.8860 | 0.886040 | exact to 4 dp |
| Weighted precision | 0.8960 | 0.896020 | exact to 4 dp |

Per-class F1 recomputed from precision and recall matches the reported F1 to 4 dp for
all five classes (Melanoma 0.9839 vs 0.9840 reported, a rounding artefact).

The residual 2.1e-06 on accuracy is explained: the recalls are given to 4 dp, so
`Σ recall_i × support_i` cannot land exactly on 648. It rounds to 648.00.

**Caveat worth repeating from §1c.1:** this confirms the reported numbers are
*mutually consistent*. It does not independently confirm them against raw predictions,
which is what §1.14 and §1b.13 did. Request the archive to close that gap.

---

## 2. Defects in the run metadata

These affected the **EfficientNet (NB5)** run. **Both are fixed in NB6** — see §2.4.
Neither ever affected a metric.

### 2.1 `timing.json` reports hardware it did not use

`timing.json` and `results_efficientnet_v2.json.timing` both carry:

> `"note": "Measured on Kaggle, one T4. Replaces the unsupported \"2xT4, 32 GB, ~6 h\" claim; the original runs were on Google Colab."`

while the same dict records `"gpu": "CPU"` and `"vram_gb": null`. The string was
hardcoded and never made conditional on `GPU_NAME`. The whole purpose of this file is
to replace an unsupported hardware claim, so it cannot ship carrying one.

**Correction (2026-10-29):** this defect was in **NB5 and NB6 only, not NB4**. NB4
builds its `timing` dict from `gpu=GPU_NAME` with no hardcoded note, so it was never
affected. An earlier draft of this file said "NB4 and NB6"; that was wrong.

**Fixed.** `gen_nb56.py` now derives the note from the device
(`f'Measured on {GPU_NAME} ({DEVICE.type})...'`) and appends an explicit
"this run was pure fp32" sentence when CUDA is absent. NB5 and NB6 regenerated;
92 recipe checks still pass, so the training recipe is unchanged.

**For the paper:** EfficientNet-B1, 25 epochs, 3.78 h on CPU. No GPU was used.

### 2.2 AMP was recorded as on but was a silent no-op

`config.use_amp = true` and `timing.amp = true`, but the training loop uses
`torch.cuda.amp.autocast(enabled=True)` and `torch.cuda.amp.GradScaler(enabled=True)`.
On CPU, `torch.cuda.amp.autocast` does not cast and `GradScaler` degrades to a
pass-through. **The run was pure fp32.**

This does not invalidate anything — fp32 is the more numerically faithful path — but
the paper must not claim mixed precision for this run.

**Fixed.** `timing.json` now records three fields instead of one: `amp_requested`
(the config flag), `amp_effective` (`use_amp and device == cuda`) and `precision`
(`'mixed (fp16 autocast + GradScaler)'` or `'fp32'`). A future run can no longer
report mixed precision it did not use.

`hyperparams_table.csv` has been rewritten accordingly: it now carries **one row per
run** (original vs retrain) with separate `amp_requested` / `amp_effective` /
`precision` / `device` columns, so the original runs' unknown hardware is marked
`unknown` rather than silently inheriting the retrain's values.

### 2.3 Operational consequence

The run consumed **zero** Kaggle GPU quota. `paper/experiments/README.md` budgets
~9 GPU-hours/week and schedules two retrains per week on that basis; that budget is
untouched.

But the risk to NB6 is larger than "likely to exceed". Extrapolating from the one
measured CPU data point (B1 fine-tune = 10.49 min/epoch at 121.5 ms/image) by the
FLOPs ratio:

| Run | GMACs vs B1 | est. CPU wall-clock | 12 h session limit |
|---|---|---|---|
| EfficientNet-B1 (measured) | 1.0× | **3.78 h** (actual) | fits |
| Swin-Tiny (NB6) | 7.63× | **~30 h** | **exceeds by 2.5×** |
| ConvNeXt-Tiny (NB4), stops at 9 ep | 7.58× | ~5.9 h | fits |
| ConvNeXt-Tiny (NB4), runs to 40 ep | 7.58× | **~34 h** | **exceeds** |

FLOP-proportional scaling ignores attention's memory-bandwidth behaviour on CPU and
the per-step overhead of a smaller batch, both of which push the real figure **up** —
so these are lower bounds. A CPU NB6 would be killed mid-run with no checkpoint
written for the final epochs.

Note NB4 is configured `num_epochs=40, freeze_epochs=3, patience=10` (changed from
30/8/5), so it has far more room to run long than the original's 9 epochs suggest.

**Fixed.** NB5 and NB6 now raise a `RuntimeError` in the setup cell if CUDA is
absent, naming the fix (Settings → Accelerator → GPU T4 ×1) and the estimated CPU
wall-clock, with an `ALLOW_CPU = False` escape hatch for deliberate smoke tests.
The guard costs seconds; discovering the problem afterwards costs the run.

**NB4 has no such guard** — it predates `gen_nb56.py`. Either add one before running
it or confirm the accelerator manually.

**Outcome: the guard worked.** NB6 ran on a Tesla T4 and finished in **0.59 h**
(§1b.10) rather than the projected ~30 h on CPU. The ~30 h estimate was never tested
against a real CPU Swin run, so treat it as what it was — an order-of-magnitude
argument for adding the guard, not a measurement. The guard is still the right thing
to keep for NB4.

### 2.4 Both defects are fixed in NB6 — verified

NB6's metadata is correct where NB5's was wrong. Confirmed by reading the delivered
`timing.json`:

| Field | NB5 (EfficientNet) | NB6 (Swin) |
|---|---|---|
| `gpu` | `"CPU"` | `"Tesla T4"` |
| `vram_gb` | `null` | `14.56` |
| `note` | **false** — claimed "one T4" on a CPU run | **correct** — "Measured on Tesla T4 (cuda) … Mixed precision active." |
| `amp` | `true` (a no-op; §2.2) | field replaced |
| `amp_requested` | — | `true` |
| `amp_effective` | — | `true` |
| `precision` | — | `"mixed (fp16 autocast + GradScaler)"` |

The hardcoded hardware string is gone — the note is now generated from `GPU_NAME` —
and the single ambiguous `amp` flag has been split into **requested** vs **effective**
plus an explicit `precision` string. For NB6 mixed precision was genuinely active
(cuda + `torch 2.11.0+cu128`), so `amp_effective: true` is a real measurement, not an
echo of the config.

**NB4 still carries the NB5-era bug.** Fix its `timing.json` cell before running, or
it will emit the same self-contradicting note.

---

## 3. The Keratosis regression — confirmed across all three architectures

Keratosis is the worst class in **all three** retrains, and it is the only class that
misses its ≥ 80% target in every one:

| Backbone | Keratosis recall | vs target | Precision | Specificity |
|---|---|---|---|---|
| EfficientNet-B1 | **61.39%** | −18.6 pp | 0.9065 | 0.9824 |
| Swin-Tiny | **68.35%** | −11.6 pp | 0.8710 | 0.9719 |
| ConvNeXt-Tiny v2 | **74.68%** | −5.3 pp | 0.9365 | 0.9859 |

Three independent architectures — a CNN, a transformer, and a modernised CNN — trained
from different pretrained weights on different hardware with different precision, land
in the same place. **This is a property of the data and the training objective, not of
any one model.** That makes it a finding to report, not a weakness to bury.

Note the ordering: Keratosis recall rises with overall accuracy (61.4 → 68.4 → 74.7 as
accuracy goes 84.7 → 86.0 → 89.1), and precision stays high throughout. A better model
recovers some of the gap but does not close it — consistent with a decision-threshold
effect rather than a capacity limit.

**It leaks into every other class, in both models.** Of 158 true Keratosis:

| Predicted as | EfficientNet-B1 | Swin-Tiny | ConvNeXt-Tiny v2 |
|---|---|---|---|
| BCC | 18 | 22 | — |
| Eczema | 23 | 17 | — |
| Melanoma | 9 | 3 | — |
| Tinea | 11 | 8 | — |
| **correct** | **97** | **108** | **118** |

ConvNeXt's 40 misses are not yet broken down by destination: cell 17 did not run, so
its confusion matrix was not transcribed from the console output. 118/158 correct is
derived from the reported recall (0.7468 × 158).

No other class disperses across all four alternatives this way.

**High precision, low recall, highest-or-near-highest specificity.** Both models only
predict Keratosis when very confident and otherwise assign the image elsewhere. This
is a decision-threshold asymmetry, not an inability to represent the class — note the
AUCs are respectable (0.9397 / 0.9436 / 0.9348), which says the *ranking* is
informative even where the argmax is not.

**It is the composite criterion working as designed.** Checkpoint selection uses
`0.4·mel_sens + 0.3·bcc_sens + 0.3·val_acc`. Keratosis enters only through the 0.3
accuracy term, diluted across five classes, while Melanoma and BCC carry 70% of the
weight directly. Keratosis also has the lowest loss weight of the five (0.1079,
§1.10) — it receives no clinical boost and is not a small class. All three saved
epochs therefore trade Keratosis recall for malignant sensitivity, and all three get
≥ 97.8% on Melanoma and ≥ 98.3% on BCC in exchange (ConvNeXt reaches **100%** BCC
recall).

**Write it as a design consequence with a stated cost.** A reviewer will find it in
Table 3 regardless; the honest framing is that the system is tuned for malignant
recall and the price is paid by one benign-adjacent class. The AUC evidence supports a
constructive follow-up sentence: a per-class threshold or a cost-sensitive decision
rule could recover Keratosis recall without retraining, and that is future work.

**The same class regressed hardest under federated learning** (70.10% → 56.70%,
`fedler.md` / `CLAUDE.md`) — so Keratosis is the weak point under centralized
training, federated training, and all three architectures. That is a three-way
cross-reference worth a sentence in the discussion.

**Caveat.** Some of this gap may be duplicate-driven. 22.28% of the test set has a
near-duplicate in train/val (NB2), and the clean-565 rescoring in NB1 will show
whether Keratosis recall moves differently from the other classes. Do not finalise the
numbers in this section until NB1 has run.

---

## 4. Corrections to `main.tex` supported by this file

| Paper claim | Replace with |
|---|---|
| EfficientNet test acc 85.11% / macro-AUC 0.9636 | **84.73% / 0.9704** (reproducible; the original was not) |
| Swin test acc 85.39% / macro-AUC 0.9683 | **85.97% / 0.9737** (reproducible; the original was not) |
| 14,498 images, 70/15/15 split | **6,655** images across 5 classes, **≈78/11/11** (5,180 / 748 / 727) |
| Kaggle 2×T4, 32 GB, ≈6 h | Per run, measured: EfficientNet **CPU, 3.78 h, 25 epochs**; Swin **1×T4, 0.59 h, 30 epochs**. One sentence cannot cover both — report per backbone. |
| Seed 32 | **Seed 42** |
| 30 epochs, identical protocol across backbones | EfficientNet ran **25 of 30** (early stop, saved epoch 19); Swin ran **30 of 30** (no early stop, saved epoch 30). Protocol is **not** identical across backbones — see `hyperparams_table.csv`. |
| Mixed-precision training | Not active for EfficientNet (§2.2); **active for Swin** (fp16 autocast + GradScaler, §1b.10). Must be stated per run. |
| BCC 0.97 / Melanoma 0.96 sensitivity (untraceable) | EfficientNet **BCC 0.9917 / Mel 0.9840** (§1.3); Swin **BCC 0.9835 / Mel 0.9840** (§1b.3) |
| MCC values 0.8178 / 0.8119 / 0.8174 / 0.6507 | EfficientNet **0.8113**, Swin **0.8252**, ConvNeXt **0.8650**; ensemble pending NB1. The claimed ConvNeXt MCC of 0.6507 is out by **+0.2143** — the largest single discrepancy in this file. |
| Eczema → Tinea 22%, Tinea → Eczema 19% | Measured per backbone: EfficientNet **15.9% / 14.8%**, Swin **11.0% / 16.5%**. Neither supports 22%. Ensemble pending NB1. |
| Swin is the strongest single backbone | **No longer true.** ConvNeXt v2 is the strongest: 89.13% vs Swin's 85.97%, and it wins four of five aggregates (§1c.5). Swin retains only macro-AUC. The +3.17 pp margin is ~5× Swin's margin over EfficientNet, so unlike that comparison this one is unlikely to be CI-fragile — but NB1's paired bootstrap on the clean 565 is still the test. |
| ConvNeXt test acc 71.35% / macro-AUC 0.9418 | **89.13% / 0.9684** (§1c.2). The original figure reflects a run that got ~1 epoch of fine-tuning, not the architecture. |
| ConvNeXt is the weak backbone, justifying ensemble complementarity | **This argument does not survive.** The weakness was a training artefact: `freeze_epochs 8`, `patience 5` and `num_epochs 30` stopped the run before the backbone had adapted (§2.3). Given `3 / 10 / 40` the same architecture is the **best** of the three. Any claim that the ensemble compensates for a weak ConvNeXt must be withdrawn, and the ablation re-argued on the NB1 numbers. |
| Ensemble gains come from correcting ConvNeXt's errors | Unsupported and now implausible — ConvNeXt has the fewest errors of the three (79 vs 102 / 111). Whether the ensemble beats ConvNeXt alone is an **open question** for NB1, not a given. |
| CLAHE, Gaussian+median denoising, artifact removal applied | None applied. Resize + ImageNet normalisation only, plus the listed train-time augmentations. |
| `main.tex:790` $\eta = 3\times10^{-4}$, $\lambda = 1\times10^{-4}$ | **lr 2e-4, weight decay 2e-4** for EfficientNet (checkpoint `config`, confirmed in `results_efficientnet_v2.json`). Swin uses lr 1e-4 / wd 3e-4, ConvNeXt 2e-4 / 3e-4 — so no single pair is correct for all three, which is the deeper problem with the sentence. |
| `main.tex:851,915` EfficientNet **84.97%** | This matches **neither** the original checkpoint (85.1124%) nor the retrain (84.7318%). See `FACT_CHECK.md` E2 — it was already untraceable before the retrain; the retrain does not explain it. Replace with **84.73%**. |
| `main.tex:798-804` "fine-tuned for 30 epochs" | EfficientNet's **original** run did complete 30 epochs (confirmed: 30 history rows), but ConvNeXt ran 9 and Swin 27, so "30 epochs" is wrong as a blanket statement. The retrains ran **25** (EfficientNet), **30** (Swin) and **up to 40** (ConvNeXt, `num_epochs` raised to 40 — exact count pending the archive). |
| `main.tex:878` "sensitivities of 0.97 and 0.96" for BCC / Melanoma | Not traceable to any checkpoint. Retrained EfficientNet gives **BCC 0.9917, Melanoma 0.9840** (§1.3). Note the paper's ordering is also inverted — it implies BCC 0.97 > Melanoma 0.96, and the measured values put BCC above Melanoma too, but at different magnitudes. |
| `main.tex:890` "Keratosis, 10% predicted as BCC" | **11.4%** (18/158) for this backbone. But the far larger story is omitted: Keratosis recall is **61.4%**, and it misclassifies into *all four* other classes (§3). The paper currently presents its weakest class as a minor confusion. |
| `src/models/CLAUDE.md:8-14` class counts (Melanoma 5,808 … Tinea 1,150; total 14,500) | These are the **14.5k-era** counts. The actual 5-class corpus is **6,655** with train distribution BCC 968 · Eczema 1,072 · Keratosis 977 · Melanoma 1,197 · Tinea 966 (§1.10) — near-balanced, not 40%/8%. |
| `hyperparams_table.csv` single row per backbone with `epochs_actually_run=30` for EfficientNet | **Rewritten**: one row per *run* (original vs retrain), with `saved_epoch`, `amp_requested`/`amp_effective`/`precision`/`device`, `best_val_acc_convention` and a `reproduces` column. |

---

## 5. What is still unmeasured

Do not write these into the paper yet:

- **Every ensemble number** — accuracy, macro-AUC, MCC, the 7-config ablation, the
  paired bootstrap CI, agreement tiers, calibration/ECE. All need NB1. **All three
  retrained checkpoints now exist**; they need collecting into one `CKPT_ROOT`.
- **Whether the ensemble beats ConvNeXt v2 alone.** This is now the central open
  question of the ablation, and it is genuinely open: the best single model is 3.17 pp
  ahead of the next, so the ensemble has to earn its place rather than inherit it.
- **Whether Swin's +1.24 pp over EfficientNet is significant** (§1b.12). The paired
  bootstrap in NB1 is the test; the overlapping CIs mean it may well not be.
- **ConvNeXt's confusion matrix, leakage and NPV** — not produced, because cell 17
  died before writing its JSON (§1c.9). NB1 regenerates all of it on the clean basis.
- **ConvNeXt's wall-clock, saved epoch and training dynamics** — the console paste
  does not carry them. Request the archive.
- **Every metric on the de-duplicated 565-image test set.** NB2 has run and the
  leakage is now quantified (22.28%), but no model has been *scored* on the clean
  subset yet — that happens in NB1. All of §1 and §1b is on the contaminated 727, so
  **both** retrains' paper-reportable accuracies are still unmeasured.
- **The leakage inflation gap** — how much the 22.28% overlap inflated each
  configuration. NB1 §15 computes it, and it is the one number that justifies the
  screening in Sec. 4.
- **Skin-tone stratified performance** (NB3, ITA proxy). The corpus has no Fitzpatrick
  labels; this will be a proxy with stated limitations, and must not be called a
  Fitzpatrick-stratified result. The paper's "Fitzpatrick III–V" coverage claim is
  unverified, and the project targets IV–VI.
- **Whether tuned ensemble weights beat equal weights** (NB1 §12, val-only, secondary).

---

## Source files

NB5 and NB6 delivered on 2026-10-29 as `/kaggle/working/` output archives. NB4's
archive has not been received — see below.

### `results.zip` — NB5, EfficientNet-B1

| File | Contents |
|---|---|
| `results_efficientnet_v2.json` | the single source of truth — config, splits, per-class, CM, full history, timing, verification |
| `history.csv` | 25 rows, per-epoch metrics + `epoch_seconds` |
| `timing.json` | wall-clock and device (**carries the false T4 note — see §2.1**) |
| `per_class_test.csv` | per-class support / recall / precision / specificity / AUC |
| `confusion_test.csv` | 5×5 counts |
| `curves_efficientnet_v2.pdf` | loss / accuracy / gap / sensitivity / composite / min-per-epoch |
| `efficientnet_b1_5cls_skin_v2.pth` | wrapped checkpoint (27.83 MB) — the one other notebooks consume |
| `best_efficientnet_b1_5cls_v2.pth` | bare `state_dict` (27.82 MB) |
| `__huggingface_repos__.json` | pretrained-weight provenance (`efficientnet_b1.ra4_e3600_r240_in1k` @ `4e4a9c0`) |

### `results (1).zip` — NB6, Swin-Tiny

| File | Contents |
|---|---|
| `results_swin_v2.json` | the single source of truth for §1b |
| `history.csv` | 30 rows, per-epoch metrics + `epoch_seconds` |
| `timing.json` | wall-clock and device — **correct**, with `amp_requested` / `amp_effective` / `precision` (§2.4) |
| `per_class_test.csv` | per-class support / recall / precision / specificity / AUC |
| `confusion_test.csv` | 5×5 counts |
| `curves_swin_v2.pdf` | loss / accuracy / gap / sensitivity / composite / min-per-epoch |
| `swin_tiny_5cls_skin_v2.pth` | wrapped checkpoint (111.0 MB, 177 tensors) — the one other notebooks consume |
| `best_swin_tiny_5cls_v2.pth` | bare `state_dict` (111.0 MB) |
| `__huggingface_repos__.json` | pretrained-weight provenance (`swin_tiny_patch4_window7_224.ms_in1k` @ `4cc3a72`) |

### NB4, ConvNeXt-Tiny — **archive not received**

§1c is built from the console output pasted on 2026-10-30, not from an archive. The
run wrote `convnext_tiny_5cls_skin_v2.pth` (186 tensors, verified bit-identical on
reload) and `cache_convnext_tiny_v2_test.npz`, plus the usual `history.csv`,
`timing.json`, `per_class_test.csv`, `confusion_test.csv` and curves PDF.
`results_convnext_v2.json` was **not** written — cell 17 raised `FileNotFoundError`
before reaching it (§1c.9).

Still needed: the archive, to re-derive §1c from raw predictions the way §1.14 and
§1b.13 were, and to recover the confusion matrix and training dynamics.

**All three archives must be uploaded as a Kaggle dataset before NB1 runs** — NB1
loads all three v2 checkpoints from `CKPT_ROOT` and asserts matching class order. The
archives share filenames (`history.csv`, `timing.json`, `confusion_test.csv`,
`per_class_test.csv`, `__huggingface_repos__.json`), so **keep each notebook's output
in its own folder** or they will overwrite each other. This is exactly what stopped
cell 17 (§1c.9): the ConvNeXt run could not see the other two checkpoints.

### The layout NB1 expects

Cell 4 resolves per-notebook subfolders first, so this works and keeps the colliding
filenames apart:

```
CKPT_ROOT/
  efficientnet/efficientnet_b1_5cls_skin_v2.pth   from results.zip
  convnext/convnext_tiny_5cls_skin_v2.pth         from NB4's output
  swin/swin_tiny_5cls_skin_v2.pth                 from results (1).zip
```

A flat layout with all three `*_v2.pth` side by side also resolves; the subfolders
only matter if the other nine files per archive come too. Both were checked against
the patched resolver, along with three layouts that must not pass silently: v2 plus a
stale original present (v2 wins), one backbone still original (refused as mixed), and
originals only (resolves, and prints `generation: ORIGINAL` so it is visible).

**`CLEAN_MANIFEST` is a second upload, separate from the checkpoints.** NB1 defaults
to `/kaggle/input/datasets/jainmanav20/skinsure-dedup/test_clean_manifest.csv` and
raises `FileNotFoundError` if it is absent — it will not quietly fall back to the
contaminated 727. NB2 wrote that file to `/kaggle/working`; it has to be uploaded as
its own dataset. Setting `CLEAN_MANIFEST = None` deliberately reports the
contaminated numbers and prints a warning, which is useful only as a cross-check
against §1/§1b, never as a result.
