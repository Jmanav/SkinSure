# FACT_CHECK.md — `paper/main.tex` vs. the SkinSure repository

Audit date: 2026-10-27
Paper audited: `paper/main.tex` (1113 lines, "Ensemble Learning for Early Detection of Skin Diseases in Indian Skin Tones")
Repo state: branch `federated-learning` @ `5a477ce`; branches present: `main`, `federated-learning` only.

> **Status note added 2026-10-29.** This file is a frozen audit of the **shipped**
> checkpoints as of 2026-10-27. Its verdicts are not revised by later work, because
> they describe what those artefacts contained. Two findings have since been resolved
> by new measurements, recorded elsewhere:
>
> | Finding | Resolution |
> |---|---|
> | **E2** EfficientNet 84.97% vs stored 85.11% | Still unexplained — the retrain gives **84.73%**, matching neither. The paper's 84.97% has no traceable source and should be replaced by the retrain figure. `paper/new_results.md` §4 |
> | **E5** ConvNeXt under-trained (9 epochs, 1 unfrozen) | Confirmed as the diagnosis; NB4 retrains with `freeze_epochs` 8→3, `patience` 5→10, `num_epochs` 30→40. Not yet run. |
> | **E6/E7** per-class sensitivity and confusion percentages unverifiable | Now measured for EfficientNet-B1: full 5×5 confusion matrix and per-class recall/precision/specificity/AUC in `paper/new_results.md` §1.3–1.7. Eczema→Tinea is **15.9%** (paper says 22%), Tinea→Eczema **14.8%** (paper 19%), Keratosis→BCC **11.4%** (paper 10%). Ensemble figures still pending NB1. |
> | **E8** test split is ≈11%, not 15% | Confirmed independently: 727/6,655 = 10.92%. A further finding supersedes the phrasing — **22.28% of that test split has a pHash near-duplicate in train/val**, so the reportable test set is **565 images**. `paper/experiments/README.md` |
>
> The central finding of this audit stands and was strengthened: the shipped weights
> do not reproduce their stored metrics, while a byte-identical retrain reaches
> 84.73% / 0.9704 and passes a save-then-verify gate. See `paper/REVISION_NOTES.md` §0.

## Evidence hierarchy used

| Level | Source | Notes |
|---|---|---|
| 1 | Saved notebook cell outputs; checkpoint contents | **No notebook in this repo has any saved output.** Checkpoint metadata is therefore the *only* level-1 evidence available. |
| 2 | Code that actually ran: `.py`, configs, notebook source | Notebook source proves code was *written*, not that it *ran*. |
| 3 | Data files and manifests on disk | `data/raw/`, `data/splits/` |
| 4 | Markdown docs | Hints only. **Never used alone to mark CONFIRMED.** |

### Critical caveat on notebook evidence

All four notebooks in `src/models/` have **zero saved outputs in every code cell**:

```
convnext_tiny_5class.ipynb : 14/14 code cells without output
efficientnet_b1_5class.ipynb : 15/15 code cells without output
ensemble_evaluation.ipynb  : 15/15 code cells without output
swin_tiny_5class.ipynb     : 14/14 code cells without output
```
(command: `python -c "<json cell/output census>"`, run 2026-10-27)

Consequently **no printed metric, table, confusion matrix or figure in the paper can be traced to a saved notebook output.** The only machine-verifiable numbers are those embedded in the three `*_5cls_skin.pth` checkpoints.

### Checkpoint inspection method

`torch.load(path, map_location="cpu", weights_only=False)`, read-only, via
`scratchpad/inspect2.py` and `inspect3.py`. No retraining, no dataset inference.

- `best_*.pth` (3 files) are **bare `state_dict`s** — no `config`, no `classes`, no metrics.
- `*_5cls_skin.pth` (3 files) are wrapped dicts with keys:
  `['classes','config','best_composite','best_val_acc','best_melanoma_sens','test_acc','macro_auc','history']`.

---

# 1. Claim-by-claim verification

## A. Architecture

| # | Paper claim | What the repo shows (evidence) | Lvl | Verdict | Suggested correction |
|---|---|---|---|---|---|
| A1 | **Jointly trained fusion network**: per-branch projection →512, concat → LayerNorm → MLP 1536→512→256 → 5 classes, GELU, dropout 0.5 (`main.tex` 611–673, 734–767, Fig. 2) | **No fusion network exists anywhere.** All six checkpoints have key prefixes exactly `['backbone','classifier']`. Probes for fusion/projection/top-level MLP return **0 non-backbone hits** in all six files. Final 2D weight is `classifier.5.weight (5, 256)` — a per-backbone head, not a fused head. The 1536-d vector, the 512-d projections and the fusion LayerNorm have **no corresponding tensors**. What exists instead: three independently trained backbones combined by a **weighted softmax average**, `ensemble_evaluation.ipynb` cell 5 (`p_ens = w_eff*p_eff + w_cnxt*p_cnxt + w_swin*p_swin`) and `src/models/ensemble.py:111-140`. | 1 | **CONTRADICTED** | Rewrite Sec. 5.1–5.3 and Fig. 2 as late (decision-level) fusion: three independently fine-tuned backbones, softmax outputs combined by weighted average. Delete the projection/concat/LayerNorm/fusion-MLP equations (Eq. 11–13) or move them to Future Work. |
| A2 | Any fusion-network code or checkpoint anywhere (any branch, any notebook) | Searched both branches (`git branch -a` → `main`, `federated-learning` only; user confirmed no other branches). `git log --all --diff-filter=D` shows **no deleted** curation/fusion/dataset file. No `.py` or notebook defines a multi-branch module. | 1,2 | **NO EVIDENCE** | — |
| A3 | Per-branch head is dropout **0.5** | Per-backbone heads use `dropout` then `dropout/2`, with dropout from config: Eff **0.35**, ConvNeXt **0.4**, Swin **0.4** (`efficientnet_b1_5class.ipynb` cell 6; ckpt `config`). Never 0.5. | 1 | **CONTRADICTED** | State 0.35 (Eff) / 0.4 (ConvNeXt, Swin), halved in the second dropout. |
| A4 | Pooled dims: EfficientNet-B1 = 1280, Swin-T = 768, ConvNeXt-T = 768 | Correct as backbone feature dims. `feat_dim = self.backbone.num_features  # 1280` (`efficientnet_b1_5class.ipynb` cell 6, line 239). ConvNeXt-T and Swin-T are 768 by timm definition. **But** these feed each backbone's own `LayerNorm→Linear(feat,256)` head — they are never projected to 512 or concatenated. | 1,2 | **PARTIAL** | Keep the dimensions; remove the claim that they are projected to a shared 512-d space and concatenated. |
| A5 | Exact timm names / ImageNet pretrained | Verified in ckpt `config.model_name`: `efficientnet_b1`, `convnext_tiny`, `swin_tiny_patch4_window7_224`. Created with `timm.create_model(..., pretrained=True, num_classes=0, drop_rate=…, drop_path_rate=…)` (cell 6 of each training notebook). | 1 | **CONFIRMED** | Add the exact timm identifiers to the paper (esp. `swin_tiny_patch4_window7_224`), which currently names none. |

## B. Training setup

| # | Paper claim | What the repo shows (evidence) | Lvl | Verdict | Suggested correction |
|---|---|---|---|---|---|
| B1 | 30 epochs (`main.tex` 801) | `num_epochs = 30` in all three configs — but it is a **ceiling, not the run length**. Early stopping on composite with `patience`. `history` array lengths show actual epochs run: **EfficientNet 30, Swin 27, ConvNeXt 9**. | 1 | **CONTRADICTED** (for ConvNeXt and Swin) | "Up to 30 epochs with early stopping (patience 6/5/5); the runs terminated at 30 (EfficientNet-B1), 27 (Swin-T) and 9 (ConvNeXt-T) epochs." |
| B2 | AdamW | `optim.AdamW` (cell 7 of each notebook). | 2 | **CONFIRMED** | — |
| B3 | lr **3e-4** | Config `lr`: EfficientNet **2e-4**, ConvNeXt **2e-4**, Swin **1e-4**. No backbone uses 3e-4. Moreover **layer-wise LR**: backbone group gets `lr * lr_decay_rate` = `lr * 0.1`; head gets `lr`. So effective backbone LRs are 2e-5/2e-5/1e-5. | 1 | **CONTRADICTED** | "Head LR 2e-4 (EfficientNet-B1, ConvNeXt-T) and 1e-4 (Swin-T), with the backbone at 0.1× the head LR." |
| B4 | weight decay **1e-4** | Config `weight_decay`: EfficientNet **2e-4**, ConvNeXt **3e-4**, Swin **3e-4**. | 1 | **CONTRADICTED** | State 2e-4 / 3e-4 / 3e-4. |
| B5 | "the paper names neither" a scheduler nor a freeze phase | Both exist and are material. **Scheduler**: `LambdaLR` with linear warmup then cosine decay (`lr_lambda`, cell 7) — `warmup_epochs` 3/3/5. **Freeze phase**: backbone frozen for epoch 1, unfrozen at `freeze_epochs+1`, optimizer+scheduler rebuilt (cell 8 training loop, lines 408–420) — `freeze_epochs` 6/8/8. | 2 | **CONTRADICTED** (by omission) | Add to Sec. 5.4: warmup (3/3/5 epochs) + cosine decay; backbone frozen then unfrozen after 6/8/8 epochs with layer-wise LR. |
| B6 | Batch sizes 16 / 12 / 8 per backbone | Exactly matches ckpt `config.batch_size`: Eff **16**, ConvNeXt **12**, Swin **8**. | 1 | **CONFIRMED** | — |
| B7 | Batch sizes "consistent with joint training" | **Logically impossible.** A jointly trained fusion network has one batch size; three different batch sizes are only possible because the three models were trained in three separate runs. The per-encoder batch size is itself evidence against A1. | 1 | **CONTRADICTED** | Present them as three separate training runs. |
| B8 | Label smoothing **0.1** | Only EfficientNet uses 0.1. ConvNeXt and Swin use **0.15** (`config.label_smooth`). | 1 | **PARTIAL** | "ε = 0.1 for EfficientNet-B1 and 0.15 for ConvNeXt-T and Swin-T." |
| B9 | Loss weights: inverse-sqrt freq × clinical factors, renormalised | Procedure confirmed exactly (`efficientnet_b1_5class.ipynb` cell 7, lines 287–301): `1/sqrt(counts)` → normalise → `×` boosts → renormalise. Boosts from config: **Melanoma 2.0, BCC 2.0, Tinea 3.0, Eczema 1.5**, Keratosis 1.0 (implicit). **The paper never states these factor values.** Numeric weights depend on the real train counts (claim D2), so with on-disk counts (BCC 968, Eczema 1072, Keratosis 977, Melanoma 1197, Tinea 966) the renormalised vector is ≈ **BCC 0.2149, Eczema 0.1532, Keratosis 0.2139, Melanoma 0.1942, Tinea 0.2238**. Not reproducible from the paper's stated counts. | 1,2 | **PARTIAL** | State the boost factors (2.0/2.0/3.0/1.5/1.0) and print the resulting weight vector; recompute once D2 is fixed. |
| B10 | WeightedRandomSampler, `1/n_k`, with replacement, used with weighted loss | Exactly as described: `class_weights_sampler = 1.0 / torch.tensor(counts)`; `WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)`; and `criterion` carries `weight=weights_tensor` (cells 5 and 7). Both mechanisms are active simultaneously. | 2 | **CONFIRMED** | — |
| B11 | Composite S = 0.4·mel_sens + 0.3·bcc_sens + 0.3·val_acc | Verbatim in code: `composite = 0.4 * mel_s + 0.3 * bcc_s + 0.3 * vl_acc` (cell 8, line 427), used as the checkpoint criterion. `best_composite` stored in all three ckpts. **However** `mel_s`/`bcc_s`/`vl_acc` are computed on the **validation** loader (`val_epoch(val_loader)`), which the paper states correctly. | 1,2 | **CONFIRMED** | — |
| B12 | Augmentations (Resize 256 + RandomCrop 224, h/v flip, rot ±30, ColorJitter(0.4,0.4,0.3,0.1), RandomAffine(translate 0.1, shear 10, scale 0.85–1.15), RandomGrayscale 0.05, ImageNet norm, RandomErasing(0.25, 0.02–0.2, 0.3–3.3)) | **Every value matches exactly** (`efficientnet_b1_5class.ipynb` cell 4, lines 154–172; identical in the other two). `Resize((256,256))` then `RandomCrop(224)`; `RandomRotation(30)`; `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)`; `RandomAffine(degrees=0, translate=(0.1,0.1), shear=10, scale=(0.85,1.15))`; `RandomGrayscale(p=0.05)`; `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`; `RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3))`. | 2 | **CONFIRMED** | Note `RandomAffine(degrees=0)` — rotation comes from the separate `RandomRotation(30)`. |
| B13 | Hardware: **Kaggle, dual NVIDIA T4, 32 GB combined** | **Contradicted.** All three training notebooks mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')` (cell 2), with `DRIVE_DATA = /content/drive/MyDrive/IPD/...`, `SAVE_DIR = /content/drive/MyDrive/IPD/models`, `ckpt_path = /content/...`. Config comments say "**8GB effective VRAM (10GB slice minus OS/CUDA overhead)**" and "**1 CPU core available**" — consistent with a single sliced GPU, not 2×T4/32 GB. A stripped comment in `efficientnet` cell 1 references an **H100** ("kills ~25% throughput on H100"). Device name is printed at runtime but **no output is saved**, so the actual GPU is unrecorded. | 1,2 | **CONTRADICTED** | Replace with Google Colab; state the actual accelerator or remove the hardware claim. The single-GPU/1-core/8 GB evidence rules out dual T4. |
| B14 | Total training time ≈ **6 h** | No timing instrumentation anywhere in any notebook; no saved output; no log. Unverifiable. | — | **NO EVIDENCE** | Remove, or re-measure and report per-backbone wall-clock. |
| B15 | Fixed random seed **32** (`main.tex` 802; also Sec. 4.3 for the split) | Every notebook sets `SEED = 42` (`efficientnet` l.57, `convnext` l.58, `swin` l.58, `ensemble_evaluation` l.46). Also note `cudnn.deterministic` is **explicitly disabled** (commented out, l.60–61), so runs are not bit-reproducible even at fixed seed. | 2 | **CONTRADICTED** | Change 32 → 42 throughout, and state that cuDNN determinism was disabled. |

## C. Preprocessing

| # | Paper claim | What the repo shows (evidence) | Lvl | Verdict | Suggested correction |
|---|---|---|---|---|---|
| C1 | **CLAHE** contrast normalisation applied | `grep -rniE "clahe\|createCLAHE"` over `src/ scripts/ api/` → **0 hits**. Not in any notebook transform. The training transform (cell 4) is resize/crop/flip/rotate/jitter/affine/grayscale/normalize/erase — no CLAHE. `src/preprocessing/color_normalization.py` implements **Shades-of-Gray**, not CLAHE, and is **off by default** (`configs/preprocessing.yaml`: `normalize_color: false`). | 1,2,3 | **CONTRADICTED** | Delete the CLAHE claim, or state it as planned/not-exercised. |
| C2 | **Gaussian + median denoising** applied | `grep -rniE "GaussianBlur\|medianBlur\|fastNlMeans\|bilateralFilter"` over `src/ scripts/ api/` → the only hit is `src/explainability/xai_summary.py:33` (a blur inside heatmap post-processing, unrelated to image preprocessing). No denoising in any training or eval transform. | 1,2 | **CONTRADICTED** | Delete. |
| C3 | **Artifact removal** (rulers, ink, vignetting) applied | Code *exists* but was **not used for training or evaluation**: `src/preprocessing/artifact_removal.py` has `_ink_mask`, `_glare_mask`, `_ruler_mask`, `remove_artifacts`. It is **off by default** (`remove_artifacts: false`) and post-dates the training notebooks; the notebooks never import `src/preprocessing`. No vignetting handler exists at all. `CLAUDE.md` additionally records it as untuned (fired on 47% of unmodified BCC images). | 2,3 | **CONTRADICTED** (as applied) | State that artifact removal is implemented but disabled and was not applied to the reported experiments; drop "vignetting". |
| C4 | Image **quality check** at acquisition; used in training/eval | Code exists: `src/preprocessing/quality_check.py` (`blur_score` = variance of Laplacian, `check_quality`); thresholds in `configs/preprocessing.yaml` (`min_short_side: 200`, `blur_reject_below: 5.0`, `blur_warn_below: 15.0`). It checks **resolution and focus only** — there is **no illumination-uniformity and no field-of-view check** as the paper claims (`main.tex` 371–374). Crucially it was **never applied to the dataset**: the notebooks load images directly via `ImageFolder` with no quality filter. | 2,3 | **PARTIAL** | Narrow to resolution + blur; state it is a deployment-path gate not applied to the training/test corpus. Remove illumination/field-of-view. |

## D. Data

| # | Paper claim | What the repo shows (evidence) | Lvl | Verdict | Suggested correction |
|---|---|---|---|---|---|
| D1 | Five sources: Fitzpatrick17k, PAD-UFES-20, DermaCon-IN, Melanoma Cancer Image Dataset, SDID | **No curation code and no provenance record of any kind.** No source column, no manifest, no per-source counts. `git log --all --diff-filter=D` shows no deleted curation script. The notebooks consume a pre-built folder tree `IPD_Datasets_Final_Split/{train,val,test}/<class>/`. Filenames hint at mixed provenance (`ISIC_0059562.jpg` in `data/splits/*.csv`; `*ProtectWyJQcm90ZWN0Il0*` strings typical of Fitzpatrick17k/DermNet scrapes) but this is inference, not evidence. | — | **NO EVIDENCE** | Either add the curation script + per-source manifest, or state sources without per-source counts and declare provenance unreconstructed. |
| D2 | Class counts BCC 3372 / Eczema 1788 / Keratosis 2380 / Melanoma 5808 / Tinea 1150, **total 14,498** | **Contradicted by the data on disk.** Census of `data/raw/IPD_Datasets_Final_Split` (train+val+test): **BCC 1208, Eczema 1365, Keratosis 1294, Melanoma 1587, Tinea 1201 → 5-class total 6,655.** Every number differs; the totals differ by **2.18×**. Direction of error is inconsistent (paper overstates Melanoma 3.7× and BCC 2.8×, but *understates* Tinea). The corpus also contains **four extra classes the paper never mentions** — Acne 285, Lichen_Planus 180, Psoriasis 996, Vitiligo 307 — for a 9-class total of 8,423. `fedler.md`/`CLAUDE.md` independently state 6,655/5 classes, agreeing with disk. The paper's imbalance narrative ("Melanoma 5,808 vs Tinea 1,150, ≈5:1", `main.tex` 473) is therefore also wrong: the real ratio is **1587:1201 ≈ 1.3:1**. | 3 | **CONTRADICTED** | Replace Table 1 with the on-disk counts (total 6,655) and rewrite Sec. 4.3's imbalance discussion: the corpus is only mildly imbalanced (1.3:1), which substantially weakens the stated rationale for the sampler + weighted loss. |
| D3 | Per-class counts **per source** | No source attribution exists at any level. | — | **NO EVIDENCE** | Remove the implication that per-source composition is known, or reconstruct it. |
| D4 | Skin-tone filtering: Fitzpatrick metadata filter + HSV filter, with thresholds | `grep -rniE "fitzpatrick\|hsv"` over the notebooks → **0 hits**. No filtering code, no thresholds, no record. The only Fitzpatrick logic in the repo is `api/schemas/patient.py`, a **runtime API input gate on self-reported patient metadata** — it never touched the dataset. | — | **NO EVIDENCE** | Remove the thresholds claim, or supply the filtering script. |
| D5 | Fitzpatrick range kept = **III–V** | Unverifiable (no filter code, no tone labels). Also note the paper's III–V conflicts with the project's own stated target of **IV–VI** (`CLAUDE.md`; `FitzpatrickType` gate) — the paper and the system disagree about which tones the work is for. | — | **NO EVIDENCE** | Reconcile III–V vs IV–VI and justify whichever is correct. |
| D6 | Removal of watermarked / multi-subject / AI-generated images | `grep -rniE "watermark\|duplicate\|phash"` → **0 hits** anywhere. No code, no log, no manifest. Note filenames in the corpus still carry `Protect...` watermark-service tokens, which is weak counter-evidence that watermarked images were removed. | — | **NO EVIDENCE** | Remove, or document the manual procedure and its scale. |
| D7 | Split **70/15/15, stratified, seed 32** | **Contradicted on all three counts.** The split is a **pre-existing folder tree** consumed by `ImageFolder`; no splitting code exists, so neither "stratified" nor any seed is evidenced. Actual on-disk proportions (5-class): train 5,180 (77.8%), val 748 (11.2%), test **727** (10.9%) — **not 70/15/15**. (9-class: 6,588/934/901 = 78.2/11.1/10.7%.) Seed in code is 42, not 32, and in any case is never used to split. | 3 | **CONTRADICTED** | State the actual proportions (≈78/11/11) and that the split was inherited as a fixed folder tree, not generated with a seed. |
| D8 | Image-level random split; near-duplicate check; patient/lesion grouping | No dedup code (`phash`/`duplicate` → 0 hits). No patient or lesion identifiers in the corpus (filenames only), so grouping is **impossible** to verify and was almost certainly not done. This is a material leakage risk: datasets like PAD-UFES-20 contain multiple images per lesion, and ISIC files can duplicate across sources. | — | **NO EVIDENCE** | Add an explicit limitation: no near-duplicate screening and no patient/lesion grouping were performed, so test metrics may be optimistic. |
| D9 | Split manifests exist | Only for the **federated-learning** work, and they are **not** the paper's split: `data/splits/global_test.csv` (904 rows) and `trainable_pool.csv` (5,131 rows), columns `path,label`, plus IID/non-IID shards. These were built by `scripts/make_federated_shards.py` for FL, totalling 6,035 — they do not reconstruct the paper's train/val/test. | 3 | **PARTIAL** | Do not cite these as the paper's manifests. Export the actual three-way split as a manifest. |
| D10 | Tone labels on test images → per-Fitzpatrick metrics possible | **No tone labels exist.** The split CSVs carry only `path,label`; the image tree carries only class folders. Per-Fitzpatrick metrics **cannot currently be computed**, which is a significant gap for a paper whose central claim is tone equity. | 3 | **NO EVIDENCE** | Acknowledge that no per-tone breakdown is possible with the present corpus; see Experiment E1 below. |

## E. Results

> **Blanket caveat:** `ensemble_evaluation.ipynb` has **no saved outputs**, so *no* ensemble number in the paper is traceable. The single-backbone numbers are traceable to checkpoint metadata, and **two of three disagree with the paper.**

| # | Paper claim | What the repo shows (evidence) | Lvl | Verdict | Suggested correction |
|---|---|---|---|---|---|
| E1 | **Ensemble** acc 85.39%, macro-AUC 0.9762, MCC 0.8178 | Untraceable. The computing cells exist (`ensemble_evaluation.ipynb` cells 6–8: `accuracy_score`, `roc_auc_score(..., multi_class='ovr', average='macro')`, `matthews_corrcoef`) but produced **no saved output**, and no ensemble checkpoint or results file was written (only `.png` figures, none of which are in the repo). Also note the notebook's test loader reads `LOCAL_ROOT/'test'` with `LOCAL_ROOT = Path('/dataset')` — a path that does not exist in this repo, so the run is unreproducible here. | — | **NO EVIDENCE** | Re-run `ensemble_evaluation.ipynb` and commit the executed notebook with outputs, plus a `results.json`. |
| E2 | **EfficientNet-B1** 84.97% / 0.9636 / 0.8119 | Checkpoint says **`test_acc = 0.8511235955` (85.11%)** and `macro_auc = 0.9636435813`. **Accuracy disagrees by 0.14 pp** (85.11 vs 84.97); AUC matches to 4 dp. MCC is **not stored** in any checkpoint → untraceable. | 1 | **CONTRADICTED** (acc), **CONFIRMED** (AUC), **NO EVIDENCE** (MCC) | Correct to 85.11%, or explain the discrepancy (e.g. a different eval run). |
| E3 | **Swin-T** 85.39% / 0.9683 / 0.8174 | Checkpoint: `test_acc = 0.8539325843` (**85.39%** ✓), `macro_auc = 0.9682684060` (**0.9683** ✓). MCC not stored. | 1 | **CONFIRMED** (acc, AUC), **NO EVIDENCE** (MCC) | — |
| E4 | **ConvNeXt-T** 71.35% / 0.9418 / 0.6507 | Checkpoint: `test_acc = 0.7134831461` (**71.35%** ✓), `macro_auc = 0.9417740628` (**0.9418** ✓). MCC not stored. | 1 | **CONFIRMED** (acc, AUC), **NO EVIDENCE** (MCC) | — |
| E5 | ConvNeXt is "about 14 points lower" — any sign of a bug, under-training, or different setup? | **Yes: clear under-training.** ConvNeXt's `history` arrays have length **9**, vs 30 (EfficientNet) and 27 (Swin) — it early-stopped at epoch 9 with `patience: 5`, and `freeze_epochs: 8`, meaning the backbone was unfrozen only at epoch 9. **It was fully fine-tuned for essentially one epoch.** Corroborating: final `train_acc` 0.7557 ≈ final `val_acc` 0.7547, `gap` 0.00093 — a near-zero train/val gap is the signature of under-fitting, not of a converged model. `best_val_acc` 0.7557 equals the last epoch's value, i.e. it was still improving when stopped. The paper's interpretation — that ConvNeXt is intrinsically weaker and "still contributes discriminative evidence" (`main.tex` 924–931) — is **not supported**; the deficit is a training artifact. | 1 | **CONTRADICTED** (interpretation) | Either re-train ConvNeXt with a larger patience / shorter freeze and report the fair number, or state explicitly that ConvNeXt early-stopped after 9 epochs (1 unfrozen) and that its 71.35% is an under-trained baseline, not an architectural result. **This materially affects the ablation in Sec. 6.4.** |
| E6 | Per-class sensitivity: BCC 0.97, Melanoma 0.96 | Not stored for the ensemble. Checkpoints store only **validation** `best_melanoma_sens`: Eff 0.9859, ConvNeXt 0.9859, Swin 0.9906 — these are val-set, single-backbone, and none equals 0.96. The ensemble per-class sensitivity cell (cell 7) has no saved output. BCC sensitivity for the ensemble is nowhere recorded. | — | **NO EVIDENCE** | Re-run and report with saved output; distinguish val from test sensitivity. |
| E7 | Confusions: Eczema→Tinea 22%, Tinea→Eczema 19%, Keratosis→BCC 10%; full confusion matrix + per-class P/R/F1 | **No confusion matrix is saved anywhere in the repo.** `ensemble_evaluation.ipynb` cells 8–9 compute `classification_report` and `confusion_matrix` and write `ensemble_confusion_matrix.png`, but the cells have no output and **the PNG is not in the repo** (`find . -iname "fig*"` → nothing; no `.png` results committed). The three specific percentages cannot be checked. | — | **NO EVIDENCE** | Commit the executed notebook and the confusion matrix as data (CSV), not only as an image. |
| E8 | Metrics computed on the 15% test split | Partly. The notebooks do evaluate on a held-out `test/` folder distinct from `val/` — correct in spirit. But the split is **≈11%, not 15%** (claim D7), and the paper's own figure caption/table do not state n. Test n = **727** images (5-class). | 3 | **PARTIAL** | State "held-out test split (n = 727, ≈10.9% of the corpus)". |
| E9 | Was the test set used for checkpoint selection, threshold tuning, or ensemble weights? | **Checkpoint selection: clean.** The composite is computed from `val_epoch(val_loader)` (cell 8), so selection used validation only — as claimed. **No threshold tuning** exists (plain `argmax`). **Ensemble weights: not tuned at all** — see E10. So there is no test-set leakage into model selection. However, because ConvNeXt early-stopped on a val-based criterion while the paper reports a *test* comparison, the ablation's framing is still affected (E5). | 1,2 | **CONFIRMED** (no leakage) | Say so explicitly — it is a strength worth stating. |
| E10 | How were the ensemble weights chosen, and on which split? | **They were never chosen.** `ensemble_evaluation.ipynb` cell 5: `WEIGHTS = (1.0, 1.0, 1.0)` with the inline comment "*Equal weights — change to (w_eff, w_cnxt, w_swin) to tune*". `configs/models.yaml` likewise sets `efficientnet: 1.0, convnext: 1.0, swin: 1.0`, and `src/models/ensemble.py:47` defaults to `(1.0, 1.0, 1.0)`. The ensemble is an **unweighted mean of softmax outputs**. No tuning was performed on any split. | 1,2 | **CONTRADICTED** (by implication) | Replace "weighted average" language with "unweighted (equal-weight) average of softmax probabilities"; note no weight search was performed. Giving an under-trained ConvNeXt a full 1/3 vote is worth discussing. |
| E11 | Figures `fig3_model_comparison.pdf`, `fig4_roc.jpg`, `fig5_confusion_matrix.pdf` exist; which code generated them | **None of the three exist in the repo** (`find . -iname "fig*" -not -path "./.git/*"` → empty). The plotting code exists in `ensemble_evaluation.ipynb` cells 9–13 but writes **differently named PNGs** to `/models`: `ensemble_model_comparison.png`, `ensemble_roc_curves.png`, `ensemble_confusion_matrix.png`, `ensemble_agreement.png`, `ensemble_sensitivity_radar.png`. The paper's filenames (`.pdf`, `.jpg`) match **no** output path in any notebook, so the submitted figures were produced or converted outside the repo and are unversioned. | 1,2 | **CONTRADICTED** / **NO EVIDENCE** | Commit the figure sources and regenerate from the executed notebook; align filenames. |
| E12 | Ablation Table 2 "under an identical training protocol" (`main.tex` 903–922) | **Contradicted.** The three branches were trained with **different** LR (2e-4/2e-4/1e-4), weight decay (2e-4/3e-4/3e-4), label smoothing (0.1/0.15/0.15), dropout (0.35/0.4/0.4), drop-path (0.15/0.2/0.2), batch size (16/12/8), warmup (3/3/5), freeze epochs (6/8/8), patience (6/5/5), and ran for different epoch counts (30/9/27). Nothing about the protocol is identical. Furthermore the "ablation" compares **independently trained models**, not branch-removals from a fused model — it is not an ablation of the proposed architecture. | 1 | **CONTRADICTED** | Delete "identical protocol"; tabulate the per-backbone hyperparameters. Re-label Table 2 as a single-backbone baseline comparison, not an ablation. A true ablation requires the pairwise-subset experiment (E3 below). |

## F. System claims (Section 3)

Method: `grep -rniE` over `api/ src/ scripts/ configs/ tests/` (excluding `venv/`), 2026-10-27. Counts are matching lines.

| # | Paper claim (`main.tex`) | Evidence | Lvl | Status |
|---|---|---|---|---|
| F1 | **Authentication with MFA** / encrypted credentials (250–256) | `MFA` 0, `multi.factor` 0, `authenticat` 0, `login` 0, `jwt\|oauth\|password` 0. `api/main.py` exposes `/health` and `/predict` with **no auth middleware** of any kind. `CLAUDE.md` also lists auth as "not yet created". | 2 | **NOT IMPLEMENTED** |
| F2 | **Medical history profile** — demographics, allergies, past diagnoses (258–265) | Partial. `api/schemas/patient.py` has `patient_age`, `patient_sex`, `patient_location`, `medical_history: list[str]`, consumed as a prompt string in `clinical_summarizer.py:69`. **No allergy field** (`allerg` → 0 hits), no longitudinal store, no fetch/generate logic, no persistence (`GET/POST /history` do not exist). | 2 | **PARTIAL** |
| F3 | **Metadata fused into the classifier** (394–404) | `SkinEnsemble.predict(image_tensor)` takes **only** an image tensor (`src/models/ensemble.py:95`); there is no metadata input, no concatenation before the head, and no fusion layer in any checkpoint (A1). Metadata reaches only the **LLM prompt**, never the classifier. **The paper already concedes this** (lines 400–404), so the architecture-level claim is honest about not being exercised — but Fig. 1's "clinical metadata → feature analysis" arrow still overstates it. | 1,2 | **NOT IMPLEMENTED** (as classifier fusion; correctly disclaimed in text) |
| F4 | **"Pattern Recognition score" quantifying structural similarity** (406–412) | `pattern.recognition\|PatternRecognition` → **0 hits**. No such score is computed or defined anywhere. The nearest artifact is `agreement_score` (fraction of backbones agreeing with the ensemble top-1, `ensemble.py:140`), which is a model-consensus measure, **not** structural similarity. | 2 | **NOT IMPLEMENTED** |
| F5 | **"Severity Assessment index"** (406–412) | Related functionality exists but not as an index: `src/agents/severity_triage.py` returns a **categorical** `TriageResult` (`routine`/`monitor`/`urgent`) from deterministic rules (melanoma ≥0.40 → urgent; conf <0.40 → block; rapid spread + infection → urgent). There is **no scalar index**. | 2 | **PARTIAL** |
| F6 | **Treatment recommendations from an evidence-based knowledge base** (414–420) | `src/agents/treatment_advisor.py` is **0 bytes** (empty stub, verified `ls -la`). `knowledge.base\|knowledge_base` → **0 hits**. The 14 `treatment` hits are the orchestrator's `treatment_advice_allowed` suppression flag — i.e. code that *withholds* advice, not code that produces it. | 2 | **NOT IMPLEMENTED** |
| F7 | **Geo-indexed specialist directory** (414–420) | `src/agents/doctor_referral.py` **0 bytes**; `src/agents/pharmacy_locator.py` **0 bytes**. `geo\|latitude\|longitude\|directory` → **0 hits**; `pharmac` → 0. The 28 `referral` hits are the `immediate_referral` boolean flag, not a directory lookup. | 2 | **NOT IMPLEMENTED** |

**Section 3 summary: of seven described subsystems, four are entirely absent, two are partial, one (metadata fusion) is absent but disclaimed.** Section 3 describes a design, not a built system, and does not say so. It should be reframed as "proposed system design" throughout, with an explicit implementation-status table.

---

# 2. Numbers I could not trace

Every number below appears in `paper/main.tex` with no saved output anywhere in the repo.

| Number | Location | Why untraceable |
|---|---|---|
| Ensemble acc **85.39%** | abstract, 849, 919 | `ensemble_evaluation.ipynb` has no saved outputs; no results file |
| Ensemble macro-AUC **0.9762** | abstract, 849, 919 | same |
| Ensemble MCC **0.8178** | 856, 919 | same; MCC stored in no checkpoint |
| EfficientNet MCC **0.8119** | 851, 915 | MCC not stored in any checkpoint |
| Swin MCC **0.8174** | 855, 916 | same |
| ConvNeXt MCC **0.6507** | 850, 917 | same |
| BCC sensitivity **0.97** | 878 | ensemble per-class cell has no output |
| Melanoma sensitivity **0.96** | 878 | same (ckpts store only *val* mel-sens: 0.9859/0.9859/0.9906) |
| Eczema→Tinea **22%** | 883 | no confusion matrix saved |
| Tinea→Eczema **19%** | 883 | same |
| Keratosis→BCC **10%** | 890 | same |
| Class counts **3372/1788/2380/5808/1150**, total **14,498** | Table 1, 446–463 | contradicted by disk census (6,655) |
| Imbalance ratio **≈5:1** | 473 | real ratio ≈1.3:1 |
| Split **70/15/15** | 468 | real ≈78/11/11 |
| Seed **32** | 469, 802 | code uses 42 |
| Training time **≈6 h** | 804 | no timing instrumentation, no logs |
| Hardware **2×T4 / 32 GB** | 798–800 | notebooks are Colab; comments say 8 GB / 1 CPU core |
| EfficientNet acc **84.97%** | 851, 915 | checkpoint says 85.11% — **mismatch**, not merely untraced |
| Fitzpatrick range **III–V** | 427 | no tone labels or filter code |
| **245 diagnoses / 5,450 images** (DermaCon-IN, 146–148) | lit review | external citation — out of audit scope, not repo-checkable |

---

# 3. Training runs found

### Run 1 — EfficientNet-B1 (Google Colab)
- **Location:** `src/models/efficientnet_b1_5class.ipynb`; artifacts `models/checkpoints/efficientNet/{best_,}efficientnet_b1_5cls_skin.pth`
- **Platform:** Colab (`from google.colab import drive` l.76; `SAVE_DIR=/content/drive/MyDrive/IPD/models`); config comments "8GB effective VRAM", "1 CPU core"; a stale comment mentions H100
- **Config:** bs 16, lr 2e-4 (backbone 2e-5), wd 2e-4, dropout 0.35, drop_path 0.15, ls 0.1, warmup 3, freeze 6, patience 6; **ran 30/30 epochs**
- **Produced:** `test_acc` 85.11%, `macro_auc` 0.9636, `best_val_acc` 87.69%, `best_melanoma_sens` 98.59%, `best_composite` 0.9486
- **Paper numbers it produced:** the EfficientNet row of Fig. 3 / Table 2 — AUC 0.9636 ✓, accuracy **mismatched** (paper 84.97% vs ckpt 85.11%)

### Run 2 — ConvNeXt-Tiny (Google Colab)
- **Location:** `src/models/convnext_tiny_5class.ipynb`; `models/checkpoints/convNext/`
- **Config:** bs 12, lr 2e-4, wd 3e-4, dropout 0.4, drop_path 0.2, ls 0.15, warmup 3, freeze 8, patience 5; **ran only 9/30 epochs** (early stop; backbone unfrozen at epoch 9 → ~1 unfrozen epoch)
- **Produced:** `test_acc` 71.35%, `macro_auc` 0.9418, `best_val_acc` 75.57%, `best_composite` 0.8867; train/val gap 0.0009 (under-fit signature)
- **Paper numbers it produced:** ConvNeXt row — 71.35% ✓ and 0.9418 ✓, but the paper's *interpretation* of the gap is wrong (see E5)

### Run 3 — Swin-Tiny (Google Colab)
- **Location:** `src/models/swin_tiny_5class.ipynb`; `models/checkpoints/swin/`
- **Config:** `swin_tiny_patch4_window7_224`, bs 8, lr 1e-4, wd 3e-4, dropout 0.4, drop_path 0.2, ls 0.15, warmup 5, freeze 8, patience 5; **ran 27/30 epochs**
- **Produced:** `test_acc` 85.39%, `macro_auc` 0.9683, `best_val_acc` 88.35%, `best_melanoma_sens` 99.06%, `best_composite` 0.9515
- **Paper numbers it produced:** Swin row — both ✓. **Note the paper's headline ensemble accuracy (85.39%) is numerically identical to Swin's**, consistent with the paper's own admission that the ensemble does not beat Swin on accuracy.

### Run 4 — Ensemble evaluation (location unknown, never evidenced)
- **Location:** `src/models/ensemble_evaluation.ipynb` — **no saved outputs**; paths (`LOCAL_ROOT=/dataset`, `SAVE_DIR=/models`) match **neither** Colab (`/content/...`) nor this repo, suggesting a third environment (possibly a Kaggle run, which would be the only support for the paper's Kaggle claim — but **nothing in the repo evidences it**)
- **Produced:** nothing retained. No ensemble checkpoint, no results file, no figure.
- **Paper numbers attributed to it:** 85.39% / 0.9762 / 0.8178, all per-class sensitivities, all confusion percentages, all three figures — **none traceable**

### No Kaggle run found
The paper states Kaggle 2×T4. Every notebook that demonstrably produced a checkpoint is **Colab**. The ensemble notebook's absolute paths hint at a different environment but carry no platform marker (no `/kaggle/input`, no `kaggle` string anywhere in the repo). **Either the Kaggle claim is wrong, or that run's notebook is not in this repo.**

---

# 4. Docs vs code mismatches

Listed only, not fixed, per instruction.

| # | File + line | What it says | What is actually true |
|---|---|---|---|
| 1 | `configs/models.yaml:7-10` | `weights: efficientnet/convnext/swin: 1.0` under the heading "ensemble weights" | Accurate, but `CLAUDE.md` (Ensemble voting section) describes "**weighted** average ... with weights from `configs/models.yaml`", implying tuned weights. They are uniform and were never tuned (`ensemble_evaluation.ipynb` cell 5 comment: "change to … to tune"). |
| 2 | `CLAUDE.md` ("ML Pipeline" table) | Lists hair removal, artifact removal, colour normalisation as pipeline steps | All three are `false` in `configs/preprocessing.yaml`; none was applied to training or evaluation. (CLAUDE.md does say "off by default" later, but the table reads as an active pipeline.) |
| 3 | `CLAUDE.md` ("Current Implementation State") | "Dataset: 6,655 images, 5 classes" | Correct for the 5 paper classes, but the corpus on disk holds **9** class folders totalling 8,423 images (Acne 285, Lichen_Planus 180, Psoriasis 996, Vitiligo 307 are undocumented). |
| 4 | `CLAUDE.md` (Severity triage) | "thresholds 0.40 in `severity_triage.py`" | Confirmed (`MELANOMA_URGENT_MIN_CONFIDENCE = 0.40`, `BLOCK_BELOW_CONFIDENCE = 0.40`). No mismatch — recorded as a spot-check that passed. |
| 5 | `CLAUDE.md` (Agentic layer table) | Treatment Advisor / Doctor Referral / Pharmacy Locator = "Stub" | Confirmed: all three are **0 bytes**. No mismatch. |
| 6 | `CLAUDE.md` / `fedler.md` | Centralized baseline "83.15% test acc, 98.32% melanoma sens" | This is the **federated-learning** EfficientNet baseline on the FL split (`data/splits/`), a *different* experiment from the paper's 85.11%. The two are easily conflated; neither doc flags the distinction. |
| 7 | `src/models/CLAUDE.md` | Referenced by root `CLAUDE.md` as holding "training details and sensitivity targets" | File does not exist (`src/CLAUDE.md` and `src/agents/CLAUDE.md` are listed as 0-byte stubs; `src/models/CLAUDE.md` is absent from the tree). |
| 8 | Root `CLAUDE.md` ("Last audited 2026-10-26") | Implies state verified against disk | The notebooks' seed (42), platform (Colab) and ConvNeXt's 9-epoch run are nowhere recorded in any doc; the docs do not mention that all notebook outputs are stripped. |

---

# 5. Experiments the paper needs that could be run from the repo

Compute estimates assume one mid-range GPU (the checkpoints imply ~8 GB was sufficient).

### E1 — Per-Fitzpatrick metrics *(blocked; highest scientific value)*
The paper's central claim is tone equity, yet **no tone labels exist** (D10). Two routes:
- **(a) ITA-based proxy labelling.** Compute Individual Typology Angle per test image (lesion-masked, following `ref7`), bin into Fitzpatrick groups, report accuracy/sensitivity per bin. *Effort: ~1 day. Compute: CPU-only, minutes for n=727.* Caveat: a proxy, must be labelled as such.
- **(b) Recover original source labels.** Re-derive provenance by matching filenames/hashes back to Fitzpatrick17k (which ships Fitzpatrick labels). *Effort: 1–2 days. Compute: negligible.* Also fixes D1/D3.

**Without one of these, the equity claim in the abstract and conclusion is unsupported by any measurement.**

### E2 — Bootstrap 95% CIs on all headline metrics
Resample the test set (n=727) with replacement 2,000× over cached softmax outputs; report CIs for accuracy, macro-AUC, MCC for all four configurations. *Effort: ~2 h. Compute: one inference pass (~2 min) + CPU resampling (~1 min).*
**This is close to mandatory here:** the ensemble-vs-Swin MCC gap is 0.8178 vs 0.8174 (Δ=0.0004) and accuracy is identical. At n=727 the CI will almost certainly subsume that difference, which supports the paper's own cautious framing and pre-empts an obvious reviewer objection.

### E3 — A real branch ablation (pairwise subsets)
Table 2 currently compares three independently trained models — not an ablation (E12). Evaluate all four non-trivial subsets — {E,C}, {E,S}, {C,S}, {E,C,S} — by averaging cached softmax outputs. *Effort: ~2 h (pure post-processing). Compute: one cached inference pass; subsets are free.* This directly tests "complementarity of errors" and would reveal whether dropping the under-trained ConvNeXt *improves* the ensemble — a likely and important result.

### E4 — Near-duplicate (pHash) check across splits
Compute perceptual hashes for all 6,655 images, flag Hamming distance ≤ 5 pairs spanning train/val/test, and report the leakage rate. *Effort: ~3 h. Compute: CPU-only, ~10 min.* Given five aggregated sources with no dedup (D8), this is the cheapest way to defend the headline accuracy. If leakage is material, metrics must be recomputed on a cleaned split.

### E5 — Re-train ConvNeXt-Tiny fairly
Re-run with `patience` raised (≥10) and `freeze_epochs` lowered (≤4) so the backbone actually fine-tunes. *Effort: ~30 min setup. Compute: ~1–2 GPU-hours.* Required before any claim about ConvNeXt's relative capability (E5) or any honest ablation.

### E6 — Record MCC and the confusion matrix as data
Extend `ensemble_evaluation.ipynb` to dump `results.json` (per-model acc/AUC/MCC, per-class P/R/F1/sensitivity) and `confusion_matrix.csv`, and **commit the executed notebook with outputs**. *Effort: ~1 h. Compute: one inference pass.* This alone converts ~11 of the untraceable numbers in §2 into level-1 evidence.

### E7 — Ensemble weight search on validation
If the paper is to retain any "weighted" language, grid-search weights on the **validation** split only and report the chosen vector. *Effort: ~2 h. Compute: cached val inference + CPU grid.* Otherwise state plainly that weights are uniform.

---

# 6. Ten-line summary of the biggest discrepancies

1. **The architecture in the paper does not exist.** No fusion network, no 512-d projections, no 1536-d concat, no fusion MLP — all six checkpoints contain only `backbone.*` + a per-model `classifier.*` ending in `(5, 256)`. The system is an **equal-weight average of three independently trained backbones' softmax outputs**.
2. **The dataset is 2.18× smaller than reported.** Disk holds **6,655** images across the 5 classes, not 14,498; every per-class count in Table 1 is wrong, and the real imbalance is ≈1.3:1, not 5:1 — which undercuts the stated rationale for the sampler and weighted loss.
3. **No ensemble number is traceable.** Every notebook has **zero saved outputs**, so 85.39% / 0.9762 / 0.8178, all per-class sensitivities and all confusion percentages rest on nothing retained in the repo.
4. **ConvNeXt's 14-point deficit is an artifact, not a finding.** It early-stopped at **epoch 9 of 30** with the backbone unfrozen only at epoch 9 (~1 fine-tuned epoch); its train/val gap of 0.0009 is a textbook under-fit signature. The ablation's conclusion about branch complementarity is unsupported.
5. **"Identical training protocol" is false.** The three runs differ in LR, weight decay, label smoothing, dropout, drop-path, batch size, warmup, freeze epochs, patience and epochs actually run.
6. **Hardware and seed are both wrong.** All three producing notebooks are **Google Colab** (`drive.mount`), with comments indicating ~8 GB VRAM and 1 CPU core — not Kaggle 2×T4/32 GB. The seed is **42**, not 32, and cuDNN determinism is explicitly disabled.
7. **The entire preprocessing section is unexercised.** CLAHE and Gaussian/median denoising appear **nowhere in the repo**; artifact removal exists but is disabled and post-dates training; the quality gate checks only resolution and blur (not illumination or field-of-view) and was never applied to the corpus.
8. **The data-curation pipeline has no evidence at all.** No code or record for the five sources, the Fitzpatrick/HSV tone filter, or watermark/multi-subject/AI-generated removal. The 70/15/15 stratified seed-32 split is actually an inherited folder tree at ≈78/11/11, with no dedup and no patient/lesion grouping.
9. **Four of seven Section-3 subsystems do not exist** (MFA auth, Pattern Recognition score, treatment knowledge base, geo-indexed directory — the last two are 0-byte stubs), and metadata is never fused into the classifier. Section 3 describes a design while reading as a built system.
10. **Two numbers are outright inconsistent with their own checkpoints:** EfficientNet's accuracy (paper 84.97% vs stored **85.11%**), and MCC for all four configurations is stored nowhere. On the credit side, **checkpoint selection is genuinely leakage-free** (composite computed on validation only), and the augmentation pipeline, batch sizes, composite formula, sampler/loss scheme and Swin/ConvNeXt accuracy-and-AUC figures all verify exactly.
