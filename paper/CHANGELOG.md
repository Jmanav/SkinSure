# CHANGELOG — `paper/` work, 2026-10-27 → 2026-10-29

Every change made under `paper/`, with the reason and the evidence behind it.

**Scope of changes outside `paper/`.** Through 2026-10-29 nothing outside `paper/`
was modified. On 2026-10-29, after the NB5 retrain delivered measured results, **one
file outside `paper/` was edited: `src/models/CLAUDE.md`** — see §10. It documented
the 14.5k-era class counts and a `best_val_acc` contract the shipped checkpoints do
not honour, both of which the retrain measured directly; leaving it stale would have
left the repo's own model documentation contradicting the paper.

`src/` code, `configs/`, `models/checkpoints/`, `api/`, `scripts/`, `tests/` and the
original `src/models/*.ipynb` remain untouched by this work. No checkpoint was
overwritten: the retrain writes `*_v2.pth` to Kaggle output, and the originals in
`models/checkpoints/*/` were only ever opened read-only with
`torch.load(..., map_location='cpu')`.

Unrelated working-tree changes sit outside this work and were left alone: the
summarizer's move to `claude-haiku-5-5` (`configs/agents.yaml`, `src/utils/config.py`,
`src/agents/clinical_summarizer.py`, `tests/unit/test_clinical_summarizer.py`, plus
the new `configs/llm.yaml` and `src/utils/costs.py`) and the deletion of
`scratch_test_image.png`. None of these are mine; they are noted here only so the
`git status` output is not mistaken for part of the paper work. They were committed
separately on 2026-10-30, ahead of this work, in `6961cdb`.

`paper/main.tex` has been **audited but not edited.**

---

## Contents

1. [Files created](#1-files-created)
2. [Files modified](#2-files-modified)
3. [Why each change was made](#3-why-each-change-was-made)
4. [The central finding: checkpoints do not reproduce](#4-the-central-finding)
5. [Corrections to my own earlier claims](#5-corrections-to-my-own-earlier-claims)
6. [Author decisions recorded](#6-author-decisions-recorded)
7. [Verification performed](#7-verification-performed)
8. [What remains unverified](#8-what-remains-unverified)
9. [Immediate next steps](#9-immediate-next-steps)
10. [2026-10-29 — NB2 and NB5 results received, docs reconciled](#10-2026-10-29--nb2-and-nb5-results-received-docs-reconciled)

---

## 1. Files created

| File | Size | Purpose |
|---|---|---|
| `paper/FACT_CHECK.md` | 44 KB | Claim-by-claim audit of `main.tex` against the repo |
| `paper/REVISION_NOTES.md` | 22 KB | Section-by-section rewrite plan, 13 sections |
| `paper/CHANGELOG.md` | this file | Change log with reasons |
| `paper/experiments/README.md` | 13 KB | Run order, Kaggle setup, output→paper mapping |
| `paper/experiments/hyperparams_table.csv` | 1 KB | Per-backbone hyperparameters read from checkpoint `config` dicts |
| `paper/experiments/01_eval_and_ablation.ipynb` | 61 KB | 35 cells (17 code). Repro gate → 7-config ablation → bootstrap CIs → calibration → figures |
| `paper/experiments/02_dedup_phash.ipynb` | 14 KB | 11 cells (7 code). pHash/dHash cross-split leakage audit |
| `paper/experiments/03_skin_tone_ita.ipynb` | 19 KB | 9 cells (5 code). ITA skin-tone proxy, tone-stratified metrics |
| `paper/experiments/04_convnext_retrain.ipynb` | 33 KB | 18 cells (9 code). ConvNeXt retrain with premature-stop fix |
| `paper/experiments/05_efficientnet_retrain.ipynb` | 42 KB | 24 cells (11 code). EfficientNet recovery retrain |
| `paper/experiments/06_swin_retrain.ipynb` | 42 KB | 24 cells (11 code). Swin recovery retrain |

## 2. Files modified

Only files created during this work were modified. In chronological order:

| File | Change | Reason |
|---|---|---|
| All 4 notebooks (then 6) | Baked in confirmed Kaggle mount paths | Author supplied the real dataset paths; removes 4 manual edits per run |
| `01_eval_and_ablation.ipynb` | +2 cells: §12b calibration, fig7 | Author approved adding ECE + reliability diagram |
| `01_eval_and_ablation.ipynb` | `results.json` gains `calibration`, `temperature_scaling`, `triage_threshold_reach` | Every number must exist in a file, not only in a plot |
| `experiments/README.md` | Kaggle dataset table replaced with confirmed paths | Paths are now known, not guessed |
| `experiments/README.md` | Run order reversed: retrains first, NB1 last | NB1 depends on checkpoints that must be regenerated |
| `experiments/README.md` | Added NB5/NB6 section | New notebooks needed documenting |

---

## 3. Why each change was made

### 3.1 Kaggle paths baked in (all notebooks)

**Was:** `DATA_ROOT = '/kaggle/input/ipd-datasets-final-split/...'` (a guess),
`CKPT_ROOT = '/kaggle/input/skinsure-checkpoints'` (a guess).

**Now:**
```python
DATA_ROOT = '/kaggle/input/datasets/jainmanav20/ipd-skin-dataset/IPD_Datasets_Final_Split'
CKPT_ROOT = '/kaggle/input/datasets/jainmanav20/checkpoints'
```

**Reason:** the author provided the real paths and confirmed the layout with `find`.
The data dataset has an `IPD_Datasets_Final_Split/` wrapper folder; the three
`*_5cls_skin.pth` files sit flat in the checkpoint dataset, matching NB1's resolver's
first candidate. Guessed slugs meant four hand-edits before every run.

**Also recorded:** only the five paper classes are present in the uploaded dataset —
Acne, Lichen_Planus, Psoriasis and Vitiligo were not uploaded — so the notebooks'
class filtering is a no-op there. Noted in the README so a future reader does not
assume the filter is dead code.

### 3.2 Calibration section added (NB1 §12b, fig7)

**Added:** ECE and MCE (10 equal-width bins) for all seven configurations;
temperature scaling fitted on validation by golden-section search on NLL and applied
to test; a reliability diagram plus confidence histogram
(`fig7_calibration.pdf`); `calibration_table.csv`; `reliability_bins_ecs.csv`; and
`triage_threshold_reach` in `results.json`.

**Reason:** `severity_triage.py` blocks below 0.40 confidence and escalates a melanoma
top-1 at or above 0.40. For a clinical-decision-support paper those thresholds are
only meaningful if the reported confidence behaves like a probability. Deep networks
are systematically overconfident, and label smoothing (0.1 for EfficientNet, 0.15 for
ConvNeXt and Swin) plus the class-weighted loss distort the scale further. The
measurement costs no GPU time because NB1 already caches every softmax vector.

**Design choices and why:**

- *Temperature fitted on `log(p_ensemble)`* — equal-weight averaging happens in
  probability space, so there is no single ensemble logit. Operating on log-probs is
  the standard treatment for an averaged probability vector.
- *Fitted on val, reported on test* — fitting on test would be leakage.
- *An assertion that accuracy is unchanged* — temperature scaling cannot alter any
  argmax. The notebook asserts this rather than claiming it, so a silent bug in the
  scaling code would fail loudly.
- *`triage_threshold_reach` reported* — turns "the safety rules work" from an
  assumption into a count of how many test images actually cross each threshold.

### 3.3 NB5 and NB6 created (the retrains)

**Reason:** the shipped EfficientNet and Swin checkpoints do not reproduce their own
stored metrics (§4 below). The training *recipe* is intact, so re-running it is the
only route to weights that match their reported numbers.

**Recipe is byte-identical to the originals** — the author chose this over
harmonising the protocols, because the goal is to reproduce ~85%, and changing the
recipe would forfeit the ability to claim that. Verified by 92 automated checks
(§7.3).

**Five deliberate additions, each with a reason:**

| Addition | Reason |
|---|---|
| Kaggle paths replace `drive.mount()` | `google.colab` does not exist on Kaggle |
| `history.csv` flushed **every** epoch | The original notebooks saved zero cell outputs, which is why the provenance of 85.11% cannot be inspected. The record must not depend on cell outputs surviving |
| `timing.json` with GPU name + wall-clock | Replaces the paper's unsupported "≈6 h on 2×T4". The producing runs were Colab |
| **Save-then-verify cell (§10)** | **This is the check whose absence cost the original weights** — see below |
| MCC computed and stored | The paper quotes MCC values that appear in no original checkpoint |

**The save-then-verify cell** reloads the checkpoint from disk, rebuilds the model,
loads with `strict=True`, then asserts:

1. every tensor is bit-identical to the model that was just evaluated
   (`torch.equal`, not `allclose`);
2. re-scoring the test set from the reloaded file reproduces `test_acc`;
3. the stored `test_acc` and the recomputed one agree to < 1e-9.

The original checkpoints would have failed assertion 1. Had this cell existed, the
overwrite would have surfaced in the original session instead of months later.

**One inconsistency preserved deliberately.** The original notebooks stored
`best_val_acc = max(history['val_acc'])` and
`best_melanoma_sens = max(history['melanoma_sensitivity'])` — maxima over *all*
epochs, which need not come from the saved epoch. For ConvNeXt the saved epoch was 4
(composite 0.8867) while `best_val_acc` 0.7557 came from epoch 8. Those keys are kept
unchanged for format compatibility, and the saved epoch's own values are recorded
alongside in a new `metrics_at_saved_epoch` key. Changing the originals' semantics
would have broken the "byte-identical recipe" claim.

### 3.4 README run order reversed

**Was:** NB1 → NB2 → NB1b → NB3, with NB4 anywhere.

**Now:** NB2 (now) → NB5 → NB6 → NB4 → NB1 → NB3 → NB1b.

**Reason:** NB1's reproduction gate compares each checkpoint against its own stored
metrics. With the current checkpoints it fails by design, and every downstream table,
CI and figure would be computed from weights that do not reproduce. The retrains must
produce usable checkpoints first. NB2 is the exception — it touches only images, never
weights, so its result is valid regardless and can run immediately, off the critical
path.

Also documented: Kaggle's ~9 GPU-hour weekly budget means two retrains fit one week
and the third the next; and **Save Version → Save & Run All (Commit)** rather than an
interactive session, since a browser disconnect is how provenance gets lost.

### 3.5 `REVISION_NOTES.md` created

**Reason:** the audit established *what* is wrong; this records *what to write
instead*, so the `main.tex` rewrite is a transcription task rather than a fresh
analysis. It also pins the four author decisions so they are not re-litigated.

Thirteen sections, largest first: the unreproducible-metrics problem; the
architecture rewrite; the protocol table; preprocessing; dataset numbers; platform;
results; calibration; de-duplication; the equity claim; system-architecture status;
external validation; and execution order.

---

## 4. The central finding

### The shipped checkpoints do not reproduce their own stored metrics

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
same images. Training accuracy is the hardest number to lose — it is measured on data
the weights were optimised against. The `model_state` in each file is therefore not
the model that produced the recorded history or the reported metrics.

### Alternative explanations, each tested and excluded

| Hypothesis | How it was excluded |
|---|---|
| Kaggle environment fault | Ran EfficientNet locally on the local test images: **66.9876%**, matching Kaggle to 4 d.p., **100% prediction agreement**, max probability difference `2e-06` |
| Wrong test set | All 727 cached filenames are exactly the local `test/` folder; 0 class disagreements; 3 train-folder filename collisions, all with differing classes (i.e. unrelated files) |
| Label permutation | Identity mapping is optimal across all 120 permutations for all three models; every cached label matches its source folder |
| Eval transform drift | Byte-identical to the original `val_tf`: `Resize((224,224))` → `ToTensor` → ImageNet normalise |
| Model class mismatch | NB1's classes match the originals line for line; 0 missing keys, 0 extra keys, 0 shape mismatches under timm 1.0.27 |
| Construction-time dropout | Proven irrelevant: identical logits for `(0.4,0.2)`, `(0.0,0.0)` and `(0.9,0.9)` under `.eval()` |
| `channels_last` difference | Original used it, NB1 does not; max logit difference `6.9e-07` |
| Weight corruption | All load `strict=True`; `best_*.pth` and wrapped files hold **bit-identical** weights |
| Head/norm misalignment | Key sets match exactly in both directions |
| Trained on the 14.5k corpus | Does not explain it: whichever corpus was used, the test images were held out, and these weights fail on *training* data |

### Most probable mechanism

The in-memory `model` was mutated between the evaluation (original cell 10, which
reloads `CFG['ckpt_path']` and scores it) and the save (original cell 12, which writes
`model.state_dict()`), while `history` and `test_acc` were carried over from the
earlier, correct evaluation. A re-executed or out-of-order cell in Colab is the
likeliest cause. The original notebooks saved **zero cell outputs**, so this cannot be
confirmed from the artifacts.

**The author's recollection of seeing 85.11% is consistent with all of this.** The
recorded history (92.5% train / 87.7% val / 85.1% test, with a sensible 4.8-point
train-val gap) describes a healthy, converged run. That run happened; its weights are
simply not in the file.

---

## 5. Corrections to my own earlier claims

Logged because each was stated confidently before being checked.

### 5.1 "Kaggle doesn't mount at `/kaggle/input/datasets/<user>/<slug>`"

**Wrong.** The author's original paths were correct. I asserted the mount point was
`/kaggle/input/<slug>` without verifying. Cost: one unnecessary round-trip.

### 5.2 "This is a bad ConvNeXt metric"

**Wrong.** After scoring only the ConvNeXt cache I concluded its stored metric was
wrong and the other two were probably fine. When the remaining two caches arrived,
**all three failed** (−18.1, −10.4, −19.6 pp). The cause is shared, not
ConvNeXt-specific. I should have waited for all three before offering a diagnosis.

### 5.3 "The 727 test images would have been in training"

**Wrong, and the author corrected it.** I argued that if the models were trained on
the 14.5k superset then the test images were seen during training. Any training run
holds out a test split, so under either corpus those images were unseen. The argument
was dropped.

### 5.4 "`ensemble.py` has a normalisation bug — `primary_conf` is clamped to 1.0"

**Wrong.** I read the arithmetic in `predict()` without checking the constructor.
`ensemble.py:75-76` already normalises:

```python
total = sum(weights)
self._weights = tuple(w / total for w in weights)
```

So the weights are `(1/3, 1/3, 1/3)`, `ensemble_probs` sums to 1.0, and
`primary_confidence` is a genuine probability. Verified: for three models giving
0.45/0.40/0.35 on a class, the result is **0.40**, not clamped. The `min(..., 1.0)` is
float32 drift protection, exactly as its comment says.

Consequences of the correction:

- Both triage thresholds work as intended. The re-capture rule is reachable; the
  melanoma rule means what it says. My claim that re-capture was "effectively
  unreachable" was false.
- **The author said "fix it" based on my incorrect premise. No change was made to
  `ensemble.py`**, because there is nothing to fix.
- What survives from the issue is the real part — normalised is not the same as
  calibrated — which is what §12b now measures.

### 5.5 A destructive mistake, and its recovery

While baking in the Kaggle paths I rewrote the notebook JSON with `json.dumps`, which
stripped the trailing newlines from every `source` array entry and collapsed all four
notebooks into single-line cells — unrunnable. `paper/` is untracked, so there was no
git copy to restore.

Recovered by regenerating from the generator scripts in the session scratchpad, then
patching the paths **in the generators** rather than the notebook JSON. Cell counts
returned to 33/11/9/18 and all verification re-passed.

A second, harmless error followed: my validator joined `source` entries with `""`
when the generator writes them without trailing newlines, which fused each cell into
one line and reported 29 phantom syntax errors. The notebooks were fine; the checker
was wrong. Fixed to join on `"\n"` when no entry ends in a newline.

**Lesson now encoded in the workflow:** edit the generators, never the generated
`.ipynb` JSON.

---

## 6. Author decisions recorded

| # | Question | Decision | Rationale given |
|---|---|---|---|
| 1 | Retrain recipe | **Byte-identical** to originals; ConvNeXt keeps its premature-stop fix | Best chance of reproducing ~85%; ConvNeXt's 9-of-30-epoch run is a defect, not a result |
| 2 | The non-existent fusion network | **Rewrite the paper as late fusion** | No GPU cost; matches the artifacts. Building it would add ~1.1 M parameters on 5,180 training images |
| 3 | Ensemble weights | **Call it equal-weight averaging**; grid search becomes a footnote | Accurate and simple; uniform untuned weights are easier to defend than tuned weights that gain little |
| 4 | Unimplemented modules | **Mark explicitly as planned/future** | Keeps the system narrative while being honest about status |
| 5 | Calibration | **Add ECE + reliability diagram** | Free (reuses cached softmax); reviewers of clinical CDS papers ask |
| 6 | 14,500-image dataset | **Not needed** | Disqualified as external validation because the 6,655 was derived from it; and the diagnosis does not depend on it, since the weights fail on training data |

### On CLAHE and Gaussian denoising — recommended against, and why

The paper claims both were applied. Neither exists in any code path
(`grep -ri clahe src/ scripts/ configs/` → 0 hits). The recommendation is to delete
the claim rather than implement it:

1. **Modality mismatch** — CLAHE is a dermoscopy convention (ISIC/HAM10000) that
   compensates for polarised-light capture. This corpus is largely clinical/consumer
   photography, where CLAHE amplifies JPEG blocking and sensor noise.
2. **It works against the paper's own goal** — CLAHE equalises local contrast, which
   normalises away the pigmentation differences the paper exists to study. `CLAUDE.md`
   already records this reasoning for Shades-of-Gray: it "can flatten genuine
   skin-tone colour, which matters for Fitzpatrick IV–VI."
3. **Train/test consistency** — the models were trained without it, so adding it at
   inference is a distribution shift that would lower accuracy.

Denoising removes the fine texture (scale, border irregularity, follicular plugging)
that dermatological diagnosis depends on, and 224×224 already discards detail.

---

## 7. Verification performed

All run locally against the real artifacts. Scripts live in the session scratchpad.

### 7.1 Checkpoint loading (`test_strict_load.py`)

Extracts the model classes from NB1 itself, so it tests the shipped code rather than a
retyped copy.

- All three checkpoints load `strict=True`: **512 / 186 / 177** tensors
- Forward `(2,3,224,224) → (2,5)`; softmax sums to 1
- Backbone feature dims **1280 / 768 / 768** as expected
- **Zero** fusion/projection/MLP tensors outside `backbone` — the evidence behind the
  architecture finding

### 7.2 Cache scoring and diagnosis

| Script | What it established |
|---|---|
| `check_cache.py` | ConvNeXt cache: 60.94% vs stored 71.35%; cache well-formed |
| `score_all.py` | All three fail; AUC implementation verified against brute force; all caches cover identical images in identical order |
| `diag_perm.py` | Identity label mapping optimal across all 120 permutations; 0/727 label-folder mismatches |
| `diag_overlap.py` | All 727 cache files are the local `test/` folder; 0 class disagreements |
| `diag_convnext.py` / `diag2.py` / `diag3.py` / `diag4.py` / `diag5.py` | Transform, model class, dropout, `channels_last`, head-norm alignment, best-vs-wrapped weights — each excluded |
| `local_eval.py` | **66.9876% locally**, 100% prediction agreement with Kaggle |
| `train_eval.py` | **62.00% on train** where history says 92.50% — the decisive result |

### 7.3 Recipe fidelity (`verify_recipe.py`) — **92 checks, 0 mismatches**

Compares NB5/NB6 against the original notebook text:

- **18 CFG keys per model** — `model_name`, `img_size`, `batch_size`, `num_epochs`,
  `lr`, `weight_decay`, `dropout`, `drop_path_rate`, `label_smooth`,
  `warmup_epochs`, `use_amp`, `freeze_epochs`, `patience`, `lr_decay_rate`, and all
  four clinical boosts (numeric equivalence, so `2e-4` matches `0.0002`)
- **Transform pipelines op by op** — `train_tf` 11 ops, `val_tf` 3 ops, both exact
- **14 verbatim recipe fragments** — `1/sqrt(count)` loss weights, each clinical
  boost, label smoothing, layer-wise backbone LR, warmup+cosine lambda, AMP autocast,
  grad clip 1.0, composite criterion, unfreeze at `freeze_epochs+1`, early stop on
  patience, weighted sampler, GELU head, `channels_last`
- **5 must-be-absent** — `google.colab`, `drive.mount`, `createCLAHE`,
  `GaussianBlur`, `medianBlur`
- **7 required additions** — `history.csv` flush, `timing.json`, `matthews_corrcoef`,
  `strict=True`, `torch.equal` tensor comparison, `REPRODUCTION FAILED` guard,
  `SEED = 42`

Whitespace is collapsed before comparison, because the two original notebooks format
the same `CrossEntropyLoss` call differently (Swin on one line). An earlier version of
this checker reported that as a mismatch; it was a formatting difference, not a recipe
difference.

### 7.4 Calibration math (`test_calib.py`)

Functions extracted from NB1 and tested against cases with known answers:

| Check | Result |
|---|---|
| Perfectly calibrated predictor (n=200,000) | ECE = 0.00224 ✓ ≈ 0 |
| Claims 95%, right 65% | ECE = 0.3005, MCE = 0.3005 ✓ ≈ 0.30 |
| Bins partition the data | 5,000/5,000, 10 bins, empty bins carry `n=0` ✓ |
| Over-sharp model | fitted T = 2.478 > 1 ✓ |
| Val NLL improves | 1.4804 → 1.1084 ✓ |
| Scaled rows still sum to 1 | ✓ |
| **argmax never changes** | ✓ (also asserted inside the notebook) |
| Test ECE improves | 0.2353 → 0.0164 ✓ |
| `apply_T(p, 1.0)` is the identity | ✓ |
| Well-matched model | T = 0.827, near 1 ✓ |

### 7.5 Notebook integrity (`verify_nbs.py`)

All six notebooks: valid JSON, nbformat 4.5, **0 syntax errors across 60 code cells**,
no collapsed cells, correct baked paths, no stale slugs.

### 7.6 NB2/NB3 logic (`test_nb23_logic.py`)

- Hamming: hex↔bits round-trip, known distances (1, 64, 63), symmetry, agreement with
  brute force on 200 random pairs, planted cross-split duplicates recovered at d=0 and
  d=3, same-split pair correctly excluded, test-contamination attribution correct
- ITA: strictly decreasing light→dark on synthetic images (70.9 / 40.6 / −10.2 /
  −62.1), correct bin assignment, all boundary cases, NaN/None→`unknown`, uniform
  image handled, corrupt file → `ok=False`/NaN without raising

---

## 8. What remains unverified

| # | Gap | Impact |
|---|---|---|
| 1 | **NB1 sections 7–15 have never executed.** This venv has no `sklearn`, `pandas`, `matplotlib` or `seaborn`, so the metric, bootstrap, figure and calibration-integration cells are syntax-checked only | Treat NB1's first run as a debug run. The calibration *math* is tested standalone, but its integration with `CONFIGS`/`config_probs` is not |
| 2 | **Whether the retrain reproduces ~85%.** The original ran with `cudnn.deterministic = False` on different hardware, so a byte-identical recipe will land *near* the original, not on it | Whatever NB5 produces becomes the paper's number |
| 3 | **`imagehash` is not installed locally**, so actual pHash/dHash values are untested — only the distance machinery around them | NB2 pip-installs it; low risk |
| 4 | **Runtime estimates are extrapolations**, not measurements, NB4's 3–4.5 h especially | `timing.json` will give the real figure |
| 5 | **The ITA bin convention** was ambiguous in the original spec ("very light > 55" vs "light 41–55"). Half-open `(lo, hi]` is used, so ITA=55 → `light` | Change the comparison in NB3's setup cell for the other convention |
| 6 | **Original dataset provenance.** The EfficientNet config comment says "smaller dataset (14.5k)" while the loader reads a tree holding 6,638 five-class images locally. Whether the Drive copy matched cannot be checked from here | Report the actual counts; describe 14.5k as the source pool if it was genuinely used upstream |
| 7 | **`ensemble_evaluation.ipynb` uses paths `/dataset` and `/models`**, matching neither Colab nor this repo, hinting at a third environment | May be where the 85.39% ensemble figure originated; unrecoverable |
| 8 | **Training wall-clock, per-source dataset composition, and the Fitzpatrick III–V vs IV–VI discrepancy** are unverifiable from code | Flagged in `REVISION_NOTES.md` §4, §5, §9 |

---

## 9. Immediate next steps

> **Superseded by §10.** Steps 1 and 2 below are **done** (2026-10-29). The current
> next step is NB6, then NB4, then NB1. Kept for the record.

1. ~~**`02_dedup_phash.ipynb`**~~ — **DONE**: 22.28% of the test set leaks into
   train/val; `test_clean_manifest.csv` keeps 565 images.
2. ~~**`05_efficientnet_retrain.ipynb`**~~ — **DONE**: 84.73% / 0.9704 / MCC 0.8113,
   save-then-verify passed. The recipe reproduces.
3. **`06_swin_retrain.ipynb`** then **`04_convnext_retrain.ipynb`** — the third goes
   in the next GPU week.
4. **`01_eval_and_ablation.ipynb`** — once all three checkpoints exist. Produces every
   table, CI, calibration number and figure the revision needs.
5. **`03_skin_tone_ita.ipynb`** and NB1 §15 — last.

Sections 1–5 of the `main.tex` rewrite (architecture, protocol, preprocessing,
dataset, platform) can begin now, since they describe the method rather than the
results. Sections 6–7 must wait for step 4.

---

## 10. 2026-10-29 — NB2 and NB5 results received, docs reconciled

Two notebooks delivered results. Both were cross-checked against their raw artefacts
before anything was written into the docs, and both produced findings that required
changes beyond simply recording a number.

### 10.1 NB2 — 22.28% test-set leakage

`02_dedup_phash.ipynb` found **162 of 727 test images (22.28%)** with a perceptual-hash
near-duplicate (pHash, Hamming d≤5) in train or val. `test_clean_manifest.csv` keeps
**565**: Keratosis 133 · Melanoma 124 · Eczema 111 · BCC 106 · Tinea 91.

This is a measurement problem, not a training problem, so the fix belongs at
evaluation time and the training pool is left intact. Consequences recorded:

- `01_eval_and_ablation.ipynb` rebuilt so the clean 565 is the **primary** basis for
  every reported metric (§3 onward), with §6's reproduction gate deliberately left on
  the full 727 — the only basis comparable to the values stored in the checkpoints —
  and §15 repurposed to report full-test numbers as a contamination appendix.
- A latent bug was found and fixed in the process: the original §15 matched the
  manifest on absolute paths, but NB2 runs under `/kaggle/working/...` while NB1
  mounts `/kaggle/input/...`, so the match would have returned **zero rows** and
  silently kept all 727. It now keys on `<ClassName>/<filename>` and raises if nothing
  matches. Six-case test suite in the scratchpad (`test_cleanmask.py`), all passing.
- Dropped from the plan as not worth the time: internal train/train dedup (an
  efficiency cost, not a validity one) and a threshold-sensitivity curve (d≤5 is a
  defensible standard cut).

### 10.2 NB5 — EfficientNet-B1 retrain: the recipe reproduces

**84.73% / macro-AUC 0.9704 / MCC 0.8113**, save-then-verify passed (512/512 tensors
bit-identical, reloaded accuracy matching to < 1e-9).

Against the original's claimed 85.11%: **−0.38 pp**, which is **2.77 images out of 727**
and **0.29 standard errors** at this sample size. Indistinguishable from noise for a
recipe that leaves `cudnn.deterministic = False`. The original figures were therefore
*achievable* — they were simply not what the shipped weights produce. That is the
central finding confirmed from the other direction.

Every number in `paper/new_results.md` §1 was independently re-derived from the raw
5×5 confusion matrix, the train distribution and `history.csv` rather than read from
the notebook's own summary fields — **161 checks across 15 groups**, of which 158
agreed exactly. Two
apparent mismatches were run down and were **not** errors: file sizes quoted in
decimal MB vs MiB, and two different `generated_utc` fields (the checkpoint's
provenance vs the results JSON's, 24 s apart).

### 10.3 Two defects in NB5's run metadata — fixed in the generator

Neither affects any metric, both would have misled the paper:

1. **`timing.json` claimed hardware it never used.** The note read "Measured on
   Kaggle, one T4" while the same dict recorded `"gpu": "CPU"`, `"vram_gb": null`.
   The string was hardcoded. The file exists specifically to replace an unsupported
   hardware claim, so it cannot ship carrying one. `gen_nb56.py` now derives the note
   from `GPU_NAME` and `DEVICE.type`.
   *Correction to an earlier draft of `new_results.md`:* the defect was in **NB5 and
   NB6 only, not NB4** — NB4 builds its note from `gpu=GPU_NAME` with no hardcoded
   string.
2. **AMP was recorded as on but was a silent no-op.** `use_amp=True` on a CPU run:
   `torch.cuda.amp.autocast` does not cast and `GradScaler` passes through, so the run
   was pure fp32. `timing.json` now records `amp_requested`, `amp_effective` and
   `precision` separately.

### 10.4 An accelerator guard added to NB5 and NB6

The CPU run also exposed an operational risk. Extrapolating from the one measured CPU
data point (B1 fine-tune 10.49 min/epoch, 121.5 ms/image) by FLOPs ratio:

| Run | GMACs vs B1 | est. CPU wall-clock | vs Kaggle's 12 h limit |
|---|---|---|---|
| EfficientNet-B1 | 1.0× | 3.78 h (measured) | fits |
| Swin-Tiny (NB6) | 7.63× | **~30 h** | **exceeds by 2.5×** |
| ConvNeXt-Tiny (NB4) to 40 ep | 7.58× | **~34 h** | **exceeds** |

NB5 and NB6 now raise a `RuntimeError` in the setup cell when CUDA is absent, naming
the fix and the estimated CPU wall-clock, with an `ALLOW_CPU = False` escape hatch.
**NB4 has no such guard** — it predates `gen_nb56.py` and needs the accelerator
checked manually. Recipe faithfulness re-verified after the edits: 92 checks, 0
mismatches.

### 10.5 A finding the retrain surfaced: Keratosis recall 61.4%

The only unmet clinical sensitivity target (≥ 80%), and it is a **consequence of the
composite checkpoint criterion** rather than a defect: Keratosis reaches the selection
score only through the 0.3 `val_acc` term diluted across five classes, while Melanoma
and BCC carry 70% of the weight directly, and it also draws the lowest loss weight
(0.1079) of the five. Precision is high (0.9065) and specificity the best of any class
(0.9824) — the model predicts Keratosis only when confident. Keratosis is also the
class that regressed hardest under federated learning (70.10% → 56.70%, `fedler.md`),
making it the weak point under both training regimes.

### 10.6 Files changed in this round

| File | Change |
|---|---|
| `paper/new_results.md` | §1.13 (`best_val_acc` convention), §1.14 (re-derivation record), §2.1–2.3 rewritten with fixes applied, §4 extended with 8 further `main.tex` discrepancies |
| `paper/experiments/01_eval_and_ablation.ipynb` | clean-565 as primary basis; path-keying bug fixed (committed `8333829`) |
| `paper/experiments/05_…ipynb`, `06_…ipynb` | device-derived timing note, three-field AMP reporting, CPU guard |
| `paper/experiments/hyperparams_table.csv` | **restructured**: one row per *run* (original vs retrain), 38 columns, adds `saved_epoch`, `amp_effective`, `precision`, `device`, `best_val_acc_convention`, `reproduces` |
| `paper/experiments/README.md` | status column, accelerator warning, step "1b" removed |
| `paper/REVISION_NOTES.md` | §0 outcome subsection, §12 execution order updated |
| `paper/FACT_CHECK.md` | status note mapping E2/E5/E6/E7/E8 to their resolutions; verdicts left frozen |
| `src/models/CLAUDE.md` | **(outside `paper/`)** actual 6,655 corpus counts beside the 14.5k-era ones, measured results against the clinical targets, the `best_val_acc` caveat for shipped checkpoints |

### 10.7 New `main.tex` discrepancies found while reconciling

Beyond those already in `FACT_CHECK.md`:

- **`main.tex:790`** states $\eta = 3\times10^{-4}$, $\lambda = 1\times10^{-4}$. The
  actual EfficientNet config is **lr 2e-4, wd 2e-4** (checkpoint `config`). Swin uses
  1e-4/3e-4 and ConvNeXt 2e-4/3e-4 — so no single pair is correct for all three,
  which is the deeper problem with the sentence.
- **`main.tex:851,915`** give EfficientNet **84.97%**, which matches neither the
  original checkpoint (85.1124%) nor the retrain (84.7318%). Still untraceable.
- **`main.tex:878`** claims BCC 0.97 / Melanoma 0.96 sensitivity; measured
  **0.9917 / 0.9840**.
- **`main.tex:890`** calls Keratosis→BCC "10%" (measured **11.4%**) while omitting
  that Keratosis recall is 61.4% and disperses into all four other classes — the
  paper presents its weakest class as a minor confusion.
- **`main.tex:798-804`** "fine-tuned for 30 epochs" is right for EfficientNet's
  original run (30 history rows, verified) but wrong as a blanket claim: ConvNeXt ran
  9 and Swin 27.
