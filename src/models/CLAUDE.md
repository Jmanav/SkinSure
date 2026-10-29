# src/models — Model Notebooks

Three Colab notebooks train one backbone each on the 5-class dark-skin dataset.
Trained `.pth` checkpoints are stored in Google Drive (`MyDrive/IPD/models/`).

## Classes and dataset

The counts below are the **14.5k-era** figures and do **not** describe the corpus the
shipped checkpoints were trained on. Kept for provenance only.

```
Melanoma             5,808  (40% — dominant but CRITICAL)
Basal_Cell_Carcinoma 3,372
Keratosis            2,380
Eczema               1,790
Tinea                1,150  (8% — smallest)
```

**Actual 5-class corpus** (measured 2026-10-29 from the NB5 retrain's
`results_efficientnet_v2.json`; see `paper/new_results.md` §1.10):

```
total 6,655 = train 5,180 | val 748 | test 727   (~78/11/11)

train distribution:
Melanoma             1,197
Eczema               1,072
Keratosis              977
Basal_Cell_Carcinoma   968
Tinea                  966
```

Near-balanced (1.24:1 max/min), **not** the 5:1 imbalance the 14.5k counts imply.
The `WeightedRandomSampler` and `1/√count` loss weighting are therefore correcting a
much milder skew than originally designed for — the clinical boosts (×2 Melanoma,
×2 BCC, ×3 Tinea, ×1.5 Eczema) now dominate the weighting, and Keratosis ends up with
the **lowest** loss weight of the five (0.1079). That is the mechanism behind the
Keratosis recall regression documented in `paper/new_results.md` §3.

The list above is ordered by size. The **model index order** is alphabetical (from `ImageFolder`) and identical in all three checkpoints: `0 Basal_Cell_Carcinoma, 1 Eczema, 2 Keratosis, 3 Melanoma, 4 Tinea`.

## Notebooks

| Notebook | Backbone | timm name | feat_dim | Head norm | Head act |
|---|---|---|---|---|---|
| `efficientnet_b1_5class.ipynb` | EfficientNet-B1 | `efficientnet_b1` | 1280 | LayerNorm | GELU |
| `convnext_tiny_5class.ipynb` | ConvNeXt-Tiny | `convnext_tiny` | 768 | LayerNorm | GELU |
| `swin_tiny_5class.ipynb` | Swin-Tiny | `swin_tiny_patch4_window7_224` | 768 | LayerNorm | GELU |

All three use identical head structure: `LayerNorm → Dropout → Linear(feat,256) → GELU → Dropout/2 → Linear(256,5)`.

## Key design decisions

### Composite checkpoint criterion (all three notebooks)
Checkpoints are saved on **composite score**, not val_acc:
```
composite = 0.4 × melanoma_sensitivity + 0.3 × bcc_sensitivity + 0.3 × val_acc
```
This ensures the saved model maximises malignant recall, not overall accuracy. A model
with 88% overall accuracy but 96% melanoma recall beats one with 90% accuracy and 88% recall.

### Class imbalance strategy (dual)
Two complementary mechanisms run simultaneously:
1. `WeightedRandomSampler` — corrects class frequencies at the batch level
2. `CrossEntropyLoss(weight=...)` — applies clinical boosts on top: Melanoma 2×, BCC 2×, Tinea 3×, Eczema 1.5×

The sampler ensures rare classes appear; the loss weight ensures the model *cares more* about getting them right.

### Freeze-then-finetune schedule
| Phase | Epochs | What trains |
|---|---|---|
| Freeze | 1 – freeze_epochs | Head only (backbone frozen) |
| Finetune | freeze_epochs+1 – end | Full model, layer-wise LR |

Backbone LR = head LR × 0.1 throughout fine-tuning. Swin uses longer freeze (8 ep) and lower base LR (1e-4) because transformers are more sensitive to early large gradient updates.

### LayerNorm head (not BatchNorm)
All three heads use `LayerNorm`. EfficientNet's original head used `BatchNorm1d`, which requires `batch_size > 1` and fails silently if `model.eval()` is not called before inference. `LayerNorm` has no batch-size constraint.

### history dict keys
All three notebooks track:
```python
history = {
    'train_loss', 'val_loss', 'train_acc', 'val_acc',
    'gap', 'melanoma_sensitivity', 'bcc_sensitivity', 'composite'
}
```

### Checkpoint keys
```python
{
    'model_state', 'classes', 'config',
    'best_composite',      # composite score of the saved epoch
    'best_val_acc',        # val_acc of the best-composite epoch
    'best_melanoma_sens',  # melanoma recall of the best-composite epoch
    'test_acc', 'macro_auc', 'history'
}
```
Load helpers use `ckpt.get('best_melanoma_sens', float('nan'))` so older checkpoints
without this key still load cleanly.

**Caveat on the three shipped checkpoints.** They do not honour the `best_val_acc`
comment above: the original notebooks end with `best_val_acc = max(history['val_acc'])`,
the maximum over *all* epochs, which is generally a different epoch from the one whose
weights were saved. ConvNeXt is the clearest case — composite selected epoch 4 while
`best_val_acc` came from epoch 8. Treat `best_val_acc` in those files as a training
statistic, not a property of the weights.

The retrain checkpoints (`*_v2.pth`, from `paper/experiments/05`, `06`, `04`) add a
`saved_epoch` key, a `metrics_at_saved_epoch` dict and a `provenance` dict, and store
`best_val_acc` as the saved epoch's own value. They also pass a save-then-verify gate:
reload with `strict=True`, `torch.equal` on all 512 tensors, and re-score to confirm
the stored `test_acc` reproduces to < 1e-9. The shipped checkpoints fail that gate —
which is why the retrains exist (`paper/REVISION_NOTES.md` §0).

## Clinical sensitivity targets

| Class | Sensitivity target | Notes |
|---|---|---|
| Melanoma | ≥ 95% | Highest — deadliest if missed |
| Basal_Cell_Carcinoma | ≥ 90% | Second most dangerous |
| Tinea | ≥ 80% | Smallest class — watch for underfitting |
| Eczema | ≥ 80% | |
| Keratosis | ≥ 80% | |

NPV for Melanoma and BCC is also reported at test time (target ≥ 0.95).

### Measured against these targets

EfficientNet-B1 retrain (NB5, 2026-10-29, n=727). Full detail and the independent
re-derivation in `paper/new_results.md` §1.3–1.4.

| Class | Recall | Target | |
|---|---|---|---|
| Basal_Cell_Carcinoma | 99.17% | ≥ 90% | ✅ |
| Melanoma | 98.40% | ≥ 95% | ✅ |
| Eczema | 82.76% | ≥ 80% | ✅ |
| Tinea | 81.74% | ≥ 80% | ✅ |
| **Keratosis** | **61.39%** | ≥ 80% | ❌ **−18.6 pp** |

NPV: Melanoma 0.9944, BCC 0.9983 — both ≥ 0.95. ✅

Keratosis is the only unmet target, and it is a **consequence of the composite
criterion**, not a bug: Keratosis reaches the selection score only through the 0.3
`val_acc` term diluted across five classes, while Melanoma and BCC carry 70% of the
weight directly, and it also receives the lowest loss weight. The saved epoch trades
Keratosis recall for malignant sensitivity. Its precision is high (0.9065) and its
specificity the best of any class (0.9824) — the model predicts Keratosis only when
confident. Keratosis is also the class that regressed hardest under federated
learning (70.10% → 56.70%, `fedler.md`), i.e. the weak point under both regimes.

Verified for the retrain: `best_val_acc` is now the **saved epoch's** value, matching
the documented contract below. The three shipped (original) checkpoints store
`max(history['val_acc'])` instead and have no `saved_epoch` key — so their
`best_val_acc` is not a property of their own weights. See
`paper/new_results.md` §1.13.

## Python modules (done)

Inference code lives in one file per backbone, plus the ensemble:
- `src/models/efficientnet.py` — `EfficientSkinClassifier` + `load_efficientnet_b1(path, device)`
- `src/models/convnext.py` — `ConvNeXtSkinClassifier` + `load_convnext_tiny(path, device)`
- `src/models/swin.py` — `SwinSkinClassifier` + `load_swin_tiny(path, device)`
- `src/models/ensemble.py` — `SkinEnsemble` + `CheckpointPaths`; weights come from `configs/models.yaml`

**Training still exists only in the notebooks.** Extracting a reusable training loop into `src/training/train_utils.py` is the first step of the federated-learning plan (`fedler.md`, Phase 0).

## agreement_score (implemented in `ensemble.py`)

`Predictions.agreement_score` = **fraction of the three models whose top-1 prediction matches the ensemble top-1**.
Values: 0.33 (one agrees), 0.67 (two agree), 1.0 (all agree). It can also be **0.0** when the weighted-average winner is not any single model's top-1.

`Predictions.confidence_tier()` maps this to `"high"` (1.0), `"moderate"` (0.67), `"low"` (0.33 or 0.0).
