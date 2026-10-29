# paper/experiments — reproducible experiments for the revised paper

These notebooks produce the evidence the revised `paper/main.tex` needs. They were
written against the audit in `paper/FACT_CHECK.md` and they describe the system that
actually exists:

> **Three independently fine-tuned backbones — EfficientNet-B1, ConvNeXt-Tiny and
> `swin_tiny_patch4_window7_224` — combined by an equal-weight average of their
> softmax outputs.** There is no jointly trained fusion network: no 512-d
> projections, no 1536-d concatenation, no fusion LayerNorm and no fusion MLP exist
> in any checkpoint.

Nothing here modifies `src/`, `configs/`, `models/checkpoints/` or the original
training notebooks. All outputs go to `/kaggle/working/`.

---

## Design rules these notebooks follow

1. **Code is the source of truth.** The model classes and the eval transform are
   copied verbatim from `src/models/*_5class.ipynb`, so all three checkpoints load
   with `strict=True`.
2. **A reproduction gate.** Notebook 1 refuses to continue unless the reproduced
   single-backbone test accuracy and macro-AUC match the values stored in the
   checkpoints to within 0.01 pp (EfficientNet 85.11 / 0.9636, ConvNeXt 71.35 /
   0.9418, Swin 85.39 / 0.9683).
3. **Every result is printed *and* written to a file.** No number exists only inside
   a plot.
4. **Five classes only**, in the order stored in the checkpoints:
   `['Basal_Cell_Carcinoma','Eczema','Keratosis','Melanoma','Tinea']`. The corpus
   also contains Acne, Lichen_Planus, Psoriasis and Vitiligo; those folders are
   ignored.
5. `SEED = 42` everywhere (the value the original runs actually used — the paper's
   "seed 32" is wrong), and `cudnn.deterministic = True` for evaluation.

---

## Kaggle datasets to attach

| Variable | Value (already baked into the notebooks) |
|---|---|
| `DATA_ROOT` | `/kaggle/input/datasets/jainmanav20/ipd-skin-dataset/IPD_Datasets_Final_Split` |
| `CKPT_ROOT` | `/kaggle/input/datasets/jainmanav20/checkpoints` |
| `OUT_DIR` | `/kaggle/working` |

Both datasets are confirmed on disk: the data dataset has a
`IPD_Datasets_Final_Split/` wrapper folder holding `{train,val,test}/<ClassName>/`,
and the three `*_5cls_skin.pth` checkpoints sit flat in the checkpoint dataset.
No path editing is needed unless the datasets are re-uploaded under new slugs.
`CKPT_ROOT` accepts either a flat folder or the repo's nested
`efficientNet/ convNext/ swin/` layout — the notebooks resolve both.

Only the five paper classes are present in the uploaded dataset; the Acne,
Lichen_Planus, Psoriasis and Vitiligo folders that exist in the local corpus were
not uploaded, so the notebooks' class filtering is a no-op here.

Upload the checkpoints from `models/checkpoints/*/`. Use the **`*_5cls_skin.pth`**
files, not `best_*.pth`: the `best_*` files are bare `state_dict`s with no `config`,
`classes` or metrics, so the reproduction gate cannot run against them.

**Accelerator:** GPU T4 x1 for Notebooks 1, 4, 5 and 6; Notebooks 2 and 3 are
CPU-only.

> **Check the accelerator before every training run.** The EfficientNet retrain of
> 2026-10-29 completed on **CPU** because the setting was left at "None". B1 was
> small enough to finish anyway (3.78 h), but it produced no mixed precision and a
> `timing.json` that claimed a T4 it never used. Swin-Tiny is ~7.6× B1's FLOPs,
> which extrapolates to **~30 h on CPU** against Kaggle's 12 h session limit — the
> session would be killed mid-run. NB5 and NB6 now raise a `RuntimeError` in the
> setup cell when CUDA is absent; **NB4 does not have that guard**, so verify it
> manually. Details in `paper/new_results.md` §2.

---

## Run order

Notebook 1 must run first — Notebooks 2 and 3 consume its cached softmax outputs,
and Notebook 1's last cell consumes Notebook 2's manifest.

> **The shipped checkpoints do not reproduce their own stored metrics**, so the
> retrains must come first. See `paper/REVISION_NOTES.md` §0 for the evidence: the
> weights score 62% on the *training* split where the checkpoint history records
> 92.5%, which is only possible if the saved `model_state` is not the model that was
> evaluated. Notebooks 04/05/06 recover usable weights, and each ends with a
> save-then-verify cell that would have caught the original failure.

| # | Notebook | HW | Runtime | Depends on | Status |
|---|---|---|---|---|---|
| 2 | `02_dedup_phash.ipynb` | CPU | **8–15 min** | data | **DONE** 2026-10-29 — 162/727 test images (22.28%) have a pHash near-duplicate (d≤5) in train/val; `test_clean_manifest.csv` keeps **565** |
| 5 | `05_efficientnet_retrain.ipynb` | T4 x1 | **3–4 h** (measured 3.78 h, but **on CPU**) | data | **DONE** 2026-10-29 — 84.73% / 0.9704 / MCC 0.8113, save-then-verify passed. See `paper/new_results.md` |
| 6 | `06_swin_retrain.ipynb` | T4 x1 | **3.5–4.5 h** (~30 h if CPU — guard now blocks this) | data | next |
| 4 | `04_convnext_retrain.ipynb` | T4 x1 | **3–4.5 h** (~4–7 min/epoch x up to 40 epochs) | data | pending — **no CPU guard**, check accelerator |
| 1 | `01_eval_and_ablation.ipynb` | T4 x1 | **12–20 min** | **all three retrained checkpoints** + data + NB2 manifest | blocked on 6 and 4 |
| 3 | `03_skin_tone_ita.ipynb` | CPU | **6–12 min** (ITA over train+test) | NB1 caches | blocked on 1 |

Notebook 1 now applies NB2's manifest from the start (`CLEAN_MANIFEST` in §1), so the
clean 565-image subset is the primary basis for every reported metric and the separate
"1b" re-run step is no longer needed. §6's reproduction gate deliberately stays on the
full 727 — that is the only basis comparable to the values stored in the checkpoints —
and §15 reports the full-test numbers as a contamination appendix.

Notebook 2 touches only images, never weights, so its result is valid regardless of
retraining — run it first, off the critical path. The three retrains are independent
of each other and of 2. Kaggle allows roughly **9 GPU-hours per week**, so two
retrains fit one week and the third goes in the next; use **Save Version → Save & Run
All (Commit)** for every training run, never an interactive session.

Notebook 1's reproduction gate checks each checkpoint against **its own** stored
metrics, so once the retrained checkpoints are in place it validates the new values
rather than the old, unreproducible ones.

> **Download `/kaggle/working/` before the session expires** — especially
> `cache_*.npz`, which Notebook 3 needs. Re-attaching the outputs of a
> finished notebook as a Kaggle dataset is the simplest way to chain sessions.

---

## Which output feeds which part of the paper

### Notebook 1 — `01_eval_and_ablation.ipynb`

| Output | Feeds |
|---|---|
| `reproduction_check.csv` | Methods: evidence that the reported checkpoints reproduce. **Fixes the EfficientNet 84.97% vs 85.11% discrepancy.** |
| `results.json` | Every number in Results. The single source of truth. |
| `ablation_table.csv` | **Table 2 (ablation)** — now a real 7-configuration subset ablation, replacing the three-independent-models table |
| `per_class_metrics.csv` | **Table 3** per-class precision / recall / F1 / specificity / AUC. Supplies the BCC 0.97 and Melanoma 0.96 sensitivities, which were previously untraceable. |
| `confusion_*_counts.csv`, `confusion_*_rownorm.csv` | Sec. 6.3 confusion analysis. Checks the claimed Eczema→Tinea 22%, Tinea→Eczema 19%, Keratosis→BCC 10%. |
| `agreement_tiers.csv` | Sec. 6 agreement analysis |
| `fig3_model_comparison.pdf` | **Fig. 3** (now with bootstrap CI error bars) |
| `fig4_roc.pdf` | **Fig. 4** |
| `fig5_confusion_matrix.pdf` | **Fig. 5** |
| `results_clean.json`, `ablation_table_clean.csv` | Sec. 6 / limitations: metrics on the de-duplicated test set |
| `cache_*.npz` | Inputs to Notebook 3 |

The **paired bootstrap** (`paired_bootstrap_ensemble_minus_swin` in `results.json`)
is the key honesty result: the paper claims macro-AUC 0.9683 → 0.9762 and MCC
0.8174 → 0.8178 over Swin. At n=727 the CI on an MCC difference of 0.0004 will
almost certainly include 0. **Where the CI includes 0, report the difference as an
observation, not an improvement** — which is what the current abstract already
hedges toward.

### Notebook 2 — `02_dedup_phash.ipynb`

| Output | Feeds |
|---|---|
| `dup_summary.json` | Sec. 4 (data) + limitations: cross-split leakage rate at d=0, ≤5, ≤8 |
| `dup_pairs.csv` | Appendix / rebuttal evidence |
| `montage_closest_pairs.png` | Manual inspection — a hash collision is not proof of duplication |
| `test_clean_manifest.csv` | Consumed by Notebook 1 section 15 |

A null result here is worth reporting: it defends the headline accuracy against the
obvious reviewer objection that five aggregated sources were never de-duplicated.

### Notebook 3 — `03_skin_tone_ita.ipynb`

| Output | Feeds |
|---|---|
| `tone_results.json` | **New subsection**: tone-stratified accuracy and per-class recall with CIs |
| `ita_per_image.csv` | Appendix; reusable per-image tone estimates |
| `fig6_tone_performance.pdf` | **Fig. 6** (new) |

This is the only measurement that speaks to the paper's equity claim, since the
corpus has **no Fitzpatrick labels**. It is a **proxy**: report it with the seven
limitations listed in the notebook's final markdown cell, and do not call it a
Fitzpatrick-stratified result. Bins with n < 30 are flagged `UNRELIABLE` and must
not be quoted.

Note the paper currently says the corpus covers Fitzpatrick **III–V** while the
project targets **IV–VI**; this notebook will show what tone range is actually
present.

### Notebooks 5 and 6 — `05_efficientnet_retrain.ipynb`, `06_swin_retrain.ipynb`

| Output | Feeds |
|---|---|
| `{efficientnet_b1,swin_tiny}_5cls_skin_v2.pth` | The checkpoints every other notebook consumes. Same wrapped format as the originals, plus `mcc`, `saved_epoch` and `metrics_at_saved_epoch` |
| `results_{efficientnet,swin}_v2.json` | Methods + Results: config, per-class metrics, confusion matrix, full history, verification record |
| `history.csv` | Training-curve evidence, flushed every epoch so it survives lost cell outputs |
| `timing.json` | Methods: **measured** wall-clock and device, replacing the unsupported "≈6 h on 2×T4". Now also records `amp_requested` / `amp_effective` / `precision`, because `torch.cuda.amp` is a silent no-op off CUDA. |
| `curves_*_v2.pdf` | Loss / accuracy / gap / sensitivity / composite / minutes-per-epoch |

The training recipe is **byte-identical** to the originals — verified by 92 automated
checks across both notebooks covering every CFG value, both transform pipelines, the
loss-weight construction, the layer-wise LR, the warmup+cosine schedule, AMP, gradient
clipping, the composite criterion and the freeze/unfreeze/early-stop logic. The only
differences are Kaggle paths, the per-epoch `history.csv`, `timing.json`, stored MCC,
and the save-then-verify cell.

**§10 is the important cell.** It reloads the checkpoint from disk, rebuilds the model,
loads `strict=True`, and asserts (1) every tensor is bit-identical to the evaluated
model, (2) re-scoring reproduces `test_acc` exactly, (3) the stored value agrees to
< 1e-9. The original checkpoints would have failed assertion 1.

### Notebook 4 — `04_convnext_retrain.ipynb`

| Output | Feeds |
|---|---|
| `history.csv` | Training-curve evidence; shows ConvNeXt actually converging |
| `timing.json` | Methods: **real** wall-clock and device, replacing the unsupported "≈6 h on 2×T4". Also records `amp_requested` / `amp_effective` / `precision`. |
| `convnext_tiny_5cls_skin_v2.pth` | Same wrapped format as the originals, plus MCC |
| `results_convnext_v2.json` | Revised Table 2 / ablation with a fairly trained ConvNeXt |

The original ConvNeXt stopped at **epoch 9 of 30** (`freeze_epochs=8` unfroze the
backbone at epoch 9, `patience=5` ended it right after), so it was fine-tuned for
roughly one epoch; its 0.0009 train/val gap is an under-fitting signature. Its
71.35% is therefore **not** an architectural result, and the current ablation
conclusion about "complementarity" rests on it. Only three settings change:
`freeze_epochs` 8→3, `patience` 5→10, `num_epochs` 30→40.

---

## Supporting file

`hyperparams_table.csv` — per-backbone hyperparameters read directly from the
checkpoint `config` dicts, for the paper's training table. It documents that the
three runs did **not** share an "identical training protocol": learning rate,
weight decay, label smoothing, dropout, drop-path, batch size, warmup, freeze
epochs, patience and epochs actually run all differ. It also records
`epochs_actually_run` (30 / 9 / 27) against `num_epochs_configured` (30).

---

## Corrections these experiments support

| Paper claim | What to write instead |
|---|---|
| Jointly trained fusion network (Sec. 5.1–5.3, Fig. 2) | Equal-weight softmax averaging over three independently fine-tuned backbones |
| "weighted average" with tuned weights | Weights are uniform (1,1,1) and were never tuned. NB1 section 12 quantifies, on val only, what tuning could buy — label it secondary. |
| lr 3e-4, wd 1e-4, ls 0.1, 30 epochs, identical protocol | Per-backbone values from `hyperparams_table.csv`; 30/9/27 epochs actually run; add the warmup+cosine schedule and the freeze phase, which the paper omits entirely |
| Kaggle 2×T4, 32 GB, ≈6 h | The producing runs were Google Colab. Report each retrain's measured `timing.json` and do not generalise across backbones. **Measured so far:** EfficientNet-B1, 25 epochs, **3.78 h on CPU, fp32** (`paper/new_results.md` §1.11). |
| Seed 32 | Seed 42 |
| 14,498 images; 70/15/15 | 6,655 across the 5 classes; actual split ≈78/11/11 (train 5,180 / val 748 / test 727), inherited as a fixed folder tree |
| CLAHE, Gaussian+median denoising, artifact removal applied | None was applied. Resize + ImageNet normalisation only (plus the listed train-time augmentations). |
| MCC values 0.8178 / 0.8119 / 0.8174 / 0.6507 | Recompute from `results.json` — MCC is stored in no checkpoint, so the current values are untraceable |
| Per-Fitzpatrick equity | ITA proxy from NB3, with its stated limitations |
