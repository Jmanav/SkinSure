# REVISION_NOTES.md — what must change in `paper/main.tex`, and why

Companion to `paper/FACT_CHECK.md` (the audit) and `paper/experiments/` (the
evidence). This file is the revision plan. **`main.tex` has not been edited.**

Decisions taken by the author, 2026-10-28:

| Question | Decision |
|---|---|
| Retrain recipe | **Byte-identical** to the originals; ConvNeXt keeps its premature-stop fix |
| Fusion-network claim | **Rewrite the paper as late fusion** — do not build the fusion network |
| Ensemble weights | **Call it equal-weight averaging**; the grid search becomes a footnote |
| Unimplemented modules | **Mark explicitly as planned/future**, keep the architecture narrative |
| Calibration | **Add ECE + reliability diagram** (Notebook 1, §12b) |

---

## 0. The blocking problem: the shipped checkpoints do not reproduce

Measured on the real test split, in **two independent environments** (Kaggle T4 and a
local CPU) that agree to four decimal places with **100% prediction agreement** and a
max probability difference of `2e-06`:

| Model | stored `test_acc` | reproduced | Δ |
|---|---|---|---|
| EfficientNet-B1 | 85.11% | **66.99%** | −18.1 pp |
| ConvNeXt-Tiny | 71.35% | **60.94%** | −10.4 pp |
| Swin-Tiny | 85.39% | **65.75%** | −19.6 pp |
| Equal-weight ensemble | 85.39% | **67.26%** | −18.1 pp |
| Ensemble macro-AUC | 0.9762 | **0.9328** | −0.0434 |

### Why this is a weights problem, not a measurement problem

The decisive evidence is the **training** split:

| Split | checkpoint `history` | measured now |
|---|---|---|
| **train** | **92.50%** | **62.00%** |
| val | 87.69% | 71.33% |
| test | 85.11% | 66.99% |
| melanoma sensitivity | 98.59% | 28.3% (on train) |

A model fitted to 92.5% on its own training images cannot later score 62% on those
same images. The `model_state` in each file is therefore **not** the model that
produced the recorded history or the reported metrics.

Ruled out, each with evidence:

- **Environment** — two machines, identical to 4 d.p., 100% prediction agreement.
- **Test set** — all 727 cached filenames are exactly the local `test/` folder; zero
  class disagreements; zero train/val contamination.
- **Label permutation** — identity mapping is optimal across all 120 permutations for
  all three models; every cached label matches its source folder.
- **Transform / model class / dropout / `channels_last`** — each tested; the eval
  transform and model class are byte-identical to the originals, and dropout is
  provably irrelevant under `.eval()`.
- **Weight corruption** — all three load `strict=True` with zero missing/extra keys
  and zero shape mismatches; the `best_*.pth` and wrapped files are bit-identical.
- **Trained on the larger 14.5k corpus** — would not explain it: whichever corpus was
  used, the test images were held out, and these weights fail on *training* data.

Most probable mechanism: the in-memory `model` was mutated between the evaluation
(original cell 10) and the save (original cell 12) — a re-executed or out-of-order
cell in Colab — while `history` and `test_acc` were carried over from the earlier,
correct evaluation. The original notebooks saved **zero cell outputs**, so the
provenance cannot be inspected.

**Consequence: every metric in the paper must come from the retrained checkpoints.**
Notebooks 05, 06 and 04 produce them; each now ends with a save-then-verify cell that
would have caught this failure immediately.

---

## 1. Architecture — Sec. 5.1–5.3 and Fig. 2 (largest rewrite)

### What the paper claims

Three backbones → three learned linear projections to 512-d → concatenate to a 1536-d
fused vector → LayerNorm *after* concatenation → Fusion MLP 1536→512→256 with GELU and
dropout 0.5 → 5 classes. Trained jointly end-to-end.

### What exists

Three **independently fine-tuned** backbones, each with its own head
(`LayerNorm → Dropout → Linear(feat,256) → GELU → Dropout → Linear(256,5)`), combined
**at inference only** by an equal-weight mean of their softmax outputs.

Evidence: every checkpoint's state dict has exactly the prefixes
`['backbone', 'classifier']`; the final 2-D weight is `classifier.5.weight` of shape
`(5, 256)`. There are **zero** projection, fusion, or 1536-d tensors in any
checkpoint. `src/models/ensemble.py:111-140` averages probabilities;
`configs/models.yaml` sets all three weights to 1.0.

| | Paper | Reality |
|---|---|---|
| Combination | jointly trained fusion network | equal-weight softmax average |
| Projections to 512-d | 3 learned | none |
| Fused 1536-d vector | yes | none |
| Fusion MLP | 1536→512→256, dropout 0.5 | none |
| Heads | one shared | three independent |
| Trainable fusion params | ~1.1 M | **0** |

### Required changes

1. **Rewrite Sec. 5.1–5.3** as *decision-level (late) fusion*: three independently
   fine-tuned backbones whose softmax outputs are averaged with equal weights.
2. **Redraw Fig. 2** — delete the projection boxes, the concatenation node, the fusion
   LayerNorm and the Fusion MLP. Show three parallel branches, each ending in its own
   5-way head, meeting at an averaging node.
3. **Delete Sec. 5.2's justification** for applying LayerNorm after concatenation
   "so that the three sub-vectors are placed on a common scale" — that layer does not
   exist.
4. **Correct the terminology throughout.** Use "soft-voting ensemble",
   "probability averaging", "late fusion" or "decision-level fusion". Do **not** use
   "feature fusion", "multi-branch", "jointly trained" or "fusion MLP".
   Sec. 2's "heterogeneous multi-branch ensemble" reads as joint training and must
   change.
5. **Reframe the contribution.** Soft voting over three ImageNet backbones is not
   novel. Lead with (a) the curated melanin-rich corpus, (b) tone-stratified
   evaluation, (c) the clinical pipeline with XAI and deterministic triage overrides.

> Averaging independently trained models **is** a legitimate ensemble — soft voting is
> a standard, well-established method. What it is not is a *fusion architecture*. The
> method is sound; only its description is wrong.

---

## 2. Training protocol — Sec. 5.4 / Table 1

### "Identical training protocol" is false in every column

| | Paper | EfficientNet | ConvNeXt | Swin |
|---|---|---|---|---|
| learning rate | 3e-4 | 2e-4 | 2e-4 | 1e-4 |
| weight decay | 1e-4 | 2e-4 | 3e-4 | 3e-4 |
| label smoothing | 0.1 | 0.1 | 0.15 | 0.15 |
| dropout | — | 0.35 | 0.4 | 0.4 |
| drop-path | — | 0.15 | 0.2 | 0.2 |
| batch size | — | 16 | 12 | 8 |
| warmup epochs | — | 3 | 3 | 5 |
| freeze epochs | — | 6 | 8 | 8 |
| patience | — | 6 | 5 | 5 |
| **epochs actually run** | 30 | 30 | **9** | 27 |

Source: `paper/experiments/hyperparams_table.csv`, generated from the checkpoint
`config` dicts. Replace the single-protocol claim with this per-backbone table.

### Components the paper omits entirely — all real, all must be added

1. **Warmup + cosine schedule** — `LambdaLR`: linear warmup for 3 (Swin: 5) epochs,
   then `0.5*(1+cos(pi*progress))` decay.
2. **Freeze-then-finetune** — backbone frozen for the first 6/8/8 epochs, then
   unfrozen with the optimiser rebuilt.
3. **Layer-wise LR decay** — backbone at `lr x 0.1`, head at `lr`.
4. **Class-weighted loss** — `1/sqrt(count)`, renormalised, then clinical boosts
   (Melanoma x2, BCC x2, Tinea x3, Eczema x1.5), renormalised again.
5. **`WeightedRandomSampler`** — running *simultaneously* with the weighted loss.
6. **Composite checkpoint criterion** —
   `0.4*melanoma_sens + 0.3*bcc_sens + 0.3*val_acc`, computed on validation only.
   **This is leakage-free and a genuine methodological strength — say so.**
7. **Gradient clipping** at 1.0, and AMP.
8. **Train-time augmentation** — RandomCrop(224) from 256, H/V flip, rotation 30°,
   ColorJitter(0.4, 0.4, 0.3, 0.1), RandomAffine(translate 0.1, shear 10,
   scale 0.85–1.15), RandomGrayscale(p=0.05), RandomErasing(p=0.25).

### ConvNeXt ran 9 of 30 epochs

`freeze_epochs=8` unfroze the backbone at epoch 9 and `patience=5` ended the run
immediately after, so ConvNeXt was fine-tuned for roughly **one** epoch. Its 0.0009
train/val gap is an under-fitting signature. **Its 71.35% is not an architectural
result**, yet the ablation's "complementarity" conclusion rests on it. Notebook 04
retrains it with `freeze_epochs` 8→3 and `patience` 5→10; state those two changes
explicitly and report the result as a corrected ConvNeXt.

---

## 3. Preprocessing — remove the CLAHE and denoising claims

The paper states CLAHE, Gaussian and median denoising, and artifact removal were
applied. **None of this exists in any code path** — `grep -ri clahe src/ scripts/
configs/` returns nothing. Preprocessing is exactly:

```
Resize((224, 224)) -> ToTensor() -> Normalize(ImageNet mean/std)
```

plus the train-time augmentations in §2.8. The only `GaussianBlur` in the repo is at
`src/explainability/xai_summary.py:33`, unrelated to the model input.

### Do not implement them either — three reasons

1. **Modality mismatch.** CLAHE is a dermoscopy convention (ISIC/HAM10000), where it
   compensates for polarised-light capture. This corpus is largely clinical/consumer
   photography, where CLAHE amplifies JPEG blocking and sensor noise.
2. **It fights the paper's own goal.** CLAHE equalises local contrast, which
   normalises away the pigmentation differences the paper exists to study.
   `CLAUDE.md` already records this reasoning for Shades-of-Gray: it "can flatten
   genuine skin-tone colour, which matters for Fitzpatrick IV–VI." The argument is
   stronger for CLAHE.
3. **Train/test consistency.** The models were trained without it; adding it at
   inference is a distribution shift that would *lower* accuracy.

Denoising removes the fine texture — scale, border irregularity, follicular plugging —
that dermatological diagnosis depends on, and 224x224 already discards detail.

**What to write instead:** resize + ImageNet normalisation only; pixel-changing steps
(hair removal, artifact removal, colour normalisation) are implemented in
`src/preprocessing/` but **off by default**, pending tuning on a held-out set. The
measured false-fire rates — 33% (hair) and 47% (artifacts) on 100 unmodified BCC
images — justify that default and are worth reporting as an honest negative result.

---

## 4. Dataset — Sec. 4

| Claim | Reality |
|---|---|
| 14,498 images | **6,655** across the five paper classes |
| 70/15/15 split | **≈78/11/11** — train 5,180 / val 748 / test 727 |
| ~5:1 imbalance | **≈1.3:1** (Melanoma 1,587 … Tinea 1,201) |
| five curated sources | no curation, dedup or filtering code exists |
| Fitzpatrick III–V | no tone labels exist in the corpus at all |

Per-class counts (5-class corpus): BCC 1,208 · Eczema 1,365 · Keratosis 1,294 ·
Melanoma 1,587 · Tinea 1,201.

Also note: the corpus on disk contains **four further classes** that the paper never
mentions — Acne 285, Lichen_Planus 180, Psoriasis 996, Vitiligo 307 (9-class total
8,423). State that the five-class subset was used, or explain the exclusion.

### On the "14.5k" figure

The EfficientNet config carries the comment `# Tightened vs 9-class version for
smaller dataset (14.5k)`, which is the likely origin of the paper's 14,498. But the
loader reads `ImageFolder(TRAIN_DIR)` over the folder tree, and that tree holds 6,638
five-class images locally. The comment and the loader disagree; **the loader is what
ran.** Report the actual counts, and if a 14.5k corpus was genuinely used upstream,
describe it as the source pool from which the 6,655 was derived.

The split is an **inherited fixed folder tree**, not a stratified random split. Say so
— and state the seed is **42**, not 32. `cudnn.deterministic` was explicitly disabled
during training, so exact bit-level reproduction is not claimed.

---

## 5. Platform and runtime — Sec. 5.4

| Claim | Reality |
|---|---|
| Kaggle, 2 x T4, 32 GB | **Google Colab** — `drive.mount('/content/drive')`, `/content/drive/MyDrive/IPD/` |
| 32 GB VRAM | config comments say "8GB effective VRAM", "1 CPU core" |
| ≈6 h training | unsupported by any artifact |

A stale comment mentioning an H100 also appears; it has no bearing on the runs.

Report the **measured** wall-clock and GPU name from each retrain's `timing.json`,
and do not generalise beyond the run that produced it.

---

## 6. Results — every number is replaced

### Must come from Notebook 1 after the retrain

- Per-backbone and ensemble accuracy, macro-AUC, MCC, balanced accuracy, macro-F1
- **Table 2** becomes a real 7-configuration subset ablation ({E}, {C}, {S}, {E,C},
  {E,S}, {C,S}, {E,C,S}), replacing the three-independent-models table
- **Table 3** per-class precision / recall / F1 / specificity / AUC — this supplies
  the BCC 0.97 and Melanoma 0.96 sensitivities, currently untraceable
- Figs. 3, 4, 5 — the files `fig3_model_comparison.pdf`, `fig4_roc.jpg` and
  `fig5_confusion_matrix.pdf` **do not exist in the repo** and must be regenerated
- Sec. 6.3 confusion claims (Eczema→Tinea 22%, Tinea→Eczema 19%, Keratosis→BCC 10%)
  must be re-derived from `confusion_*_rownorm.csv`

**MCC is stored in no checkpoint**, so the quoted 0.8178 / 0.8119 / 0.8174 / 0.6507
are untraceable. Notebooks 04/05/06 now compute and store it.

### Report confidence intervals, and do not call a null difference an improvement

Notebook 1 computes 2,000-resample bootstrap 95% CIs for every configuration and a
**paired** bootstrap of ensemble-minus-Swin with an explicit `ci_includes_zero` flag.

The paper claims macro-AUC 0.9683 → 0.9762 and MCC 0.8174 → 0.8178 over Swin alone.
At n=727 a CI on an MCC difference of **0.0004** will almost certainly include zero.
**Where the CI includes zero, report the difference as an observation, not an
improvement.**

### Ensemble weights — equal, and never tuned

`configs/models.yaml` sets all three to 1.0, and `SkinEnsemble.__init__` normalises
them to (1/3, 1/3, 1/3). Describe the system as an **equal-weight** average, not a
"weighted average with tuned weights". Notebook 1 §12 quantifies, **on validation
only**, what tuning could buy; carry it as a clearly-labelled secondary result or a
footnote.

---

## 7. Calibration — new, short subsection

Notebook 1 §12b reports ECE and MCE (10 bins) for all seven configurations,
a reliability diagram (`fig7_calibration.pdf`), and temperature scaling fitted on
validation and applied to test.

Why it belongs in a clinical-decision-support paper: `severity_triage.py` blocks
below 0.40 confidence and escalates a melanoma top-1 at or above 0.40, so those
thresholds are only meaningful if the reported confidence behaves like a probability.

Two things to state precisely:

- **The ensemble probability is correctly normalised.** `SkinEnsemble.__init__`
  divides the weights by their sum, so the averaged vector sums to 1 and
  `primary_confidence` is a genuine probability. (The `min(..., 1.0)` clamp in
  `predict()` guards float32 drift only.)
- **Normalised is not the same as calibrated.** Deep networks are systematically
  overconfident, and label smoothing (0.1/0.15) plus the class-weighted loss distort
  the probability scale further. Report the measured ECE, and note that temperature
  scaling changes no prediction — accuracy, macro-AUC and MCC are identical before and
  after — so it costs nothing to apply.

`results.json` also carries `triage_threshold_reach`: how many test images actually
fall below 0.40 and how many melanoma top-1 predictions clear it. That turns an
assumption about the safety rules into a measurement.

---

## 8. De-duplication — Sec. 4 and limitations

No dedup code exists (`phash`, `duplicate` → 0 hits), and the corpus aggregates
several sources, so a reviewer will ask about cross-split leakage.

Notebook 2 reports cross-split near-duplicate counts at Hamming distance 0, ≤5 and ≤8,
broken down by split pair and class, with a montage for visual confirmation and a
`test_clean_manifest.csv`. Notebook 1 §15 then recomputes every metric on the
de-duplicated test set (`results_clean.json`).

**A null result here is worth reporting** — it defends the headline accuracy against
the obvious objection. If duplicates *are* found, report both the full and clean
metrics.

---

## 9. The equity claim — the paper's weakest link

The framing is Fitzpatrick IV–VI, but **the corpus carries no tone labels**, so no
Fitzpatrick-stratified result is possible. The paper additionally says the corpus
covers Fitzpatrick **III–V** while the system targets **IV–VI** — an internal
contradiction that must be resolved.

Notebook 3 provides the only available evidence: ITA computed on Otsu-masked
non-lesion skin, binned into the six standard categories, with per-bin accuracy and
per-class recall and bootstrap CIs.

Non-negotiable conditions on how it is reported:

1. Call it an **ITA-based proxy**, never a Fitzpatrick-stratified result.
2. Reproduce the seven stated limitations from the notebook's final markdown cell.
3. **Never quote a bin with n < 30** — those are flagged `UNRELIABLE`.
4. If the ITA distribution turns out to be concentrated in lighter bins, that is a
   finding about the corpus and must be stated plainly, not buried.

`CLAUDE.md` also lists "Fairness is a first-class CI check" as a design invariant,
while `tests/fairness/` does not exist. Either build it or drop the claim.

---

## 10. System architecture — Sec. 3, mark unbuilt modules as planned

Per the author's decision, keep the architecture narrative but label clearly what is
not yet implemented. Current state, verified against the files:

| Sec. 3 claim | Status |
|---|---|
| User authentication, MFA | **NOT IMPLEMENTED** — 0 grep hits for MFA/auth/login/jwt/oauth/password |
| Medical History Profile | **PARTIAL** — `PatientMetadata` exists; no allergy field |
| Metadata combined with visual features at classification | **NOT IMPLEMENTED** — `predict()` takes only an image tensor (the paper already disclaims this at line 403; make it unambiguous) |
| "Pattern Recognition score" | **NOT IMPLEMENTED** — no such quantity exists |
| "Severity Assessment index" | **PARTIAL** — `TriageResult` is categorical, not a scalar index |
| Treatment knowledge base | **NOT IMPLEMENTED** — `treatment_advisor.py` is 0 bytes |
| Geo-indexed doctor/pharmacy directory | **NOT IMPLEMENTED** — both files 0 bytes |
| Grad-CAM explainability | **IMPLEMENTED** |
| Clinical summarizer (LLM) | **IMPLEMENTED**, never run against the real API |
| Severity triage with hard overrides | **IMPLEMENTED** |
| Orchestrator + `/predict`, `/health` | **IMPLEMENTED** |

Recommended wording: one explicit sentence per unbuilt module — "the treatment
advisory and referral modules are specified here and left to future work" — and a
status column in the Fig. 1 caption or an accompanying table. Do not describe unbuilt
components in the present tense.

Also soften two XAI claims: the heatmap → `XAISummary` bridge is **heuristic and
unvalidated** (thresholds are first guesses; the lesion outline is brightness
segmentation, not a trained segmenter), and Grad-CAM vs SHAP faithfulness has **not
been measured** — `scripts/evaluate_xai.py` exists but has not been run on a held-out
set.

---

## 11. External validation (optional, strongest single addition)

Single corpus, single split is a standard criticism. The author's 14.5k-image dataset
could serve as external validation **only if genuinely disjoint** from the training
split. Since the 6,655 was derived from it, it is a superset, so:

- It is **not** external validation — the test images are drawn from it.
- It could serve as a **larger training corpus**, which is a different contribution.

Before using it either way, run the Notebook 2 hashing against the current
`train/`+`val/` to measure the actual overlap. If it cannot be used, state
single-corpus evaluation as a limitation.

---

## 12. Execution order

| Step | Notebook | HW | Time | Produces |
|---|---|---|---|---|
| 1 | `02_dedup_phash` | CPU | 15 min | leakage answer — **runnable now**, independent of retraining |
| 2 | `05_efficientnet_retrain` | T4 | 3–4 h | first proof the recipe reproduces |
| 3 | `06_swin_retrain` | T4 | 3.5–4.5 h | second checkpoint |
| 4 | `04_convnext_retrain` | T4 | 3–4.5 h | corrected ConvNeXt (next GPU week) |
| 5 | `01_eval_and_ablation` | T4 | 20 min | all tables, CIs, ECE, figures |
| 6 | `03_skin_tone_ita` | CPU | 12 min | the ITA equity proxy |
| 7 | `01` §15 | CPU | 4 min | clean-test metrics |

Kaggle allows roughly 9 GPU-hours per week, so steps 2–3 fit one week and step 4 the
next. Use **Save & Run All (Commit)** for every training run.

**Do not start the `main.tex` rewrite before step 5.** Sections 6–7 depend entirely on
numbers that do not exist yet. Sections 1–5 (architecture, protocol, preprocessing,
dataset, platform) can be rewritten now, since they describe the method rather than the
results.

---

## 13. Honest summary of the revision

What survives unchanged: the clinical motivation, the literature review, the
system-pipeline concept, the Grad-CAM explainability layer, the deterministic triage
overrides (a genuine safety-engineering contribution), and the leakage-free composite
checkpoint criterion.

What must change: the architecture section, the protocol table, the preprocessing
claims, every dataset number, the platform, and every reported metric.

The method is defensible. A reproducible late-fusion ensemble at whatever accuracy the
retrain yields, reported with confidence intervals, a de-duplication audit, a
calibration analysis and an ITA-based tone proxy, is a publishable Q2 paper. An
unreproducible 85.39% attributed to an architecture that was never built is not.
