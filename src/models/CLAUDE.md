# src/models — Model Notebooks

Three Colab notebooks train one backbone each on the 5-class dark-skin dataset.
Trained `.pth` checkpoints are stored in Google Drive (`MyDrive/IPD/models/`).

## Classes and dataset

```
Melanoma             5,808  (40% — dominant but CRITICAL)
Basal_Cell_Carcinoma 3,372
Keratosis            2,380
Eczema               1,790
Tinea                1,150  (8% — smallest)
```

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

## Clinical sensitivity targets

| Class | Sensitivity target | Notes |
|---|---|---|
| Melanoma | ≥ 95% | Highest — deadliest if missed |
| Basal_Cell_Carcinoma | ≥ 90% | Second most dangerous |
| Tinea | ≥ 80% | Smallest class — watch for underfitting |
| Eczema | ≥ 80% | |
| Keratosis | ≥ 80% | |

NPV for Melanoma and BCC is also reported at test time (target ≥ 0.95).

## Converting to .py for ensemble

When moving notebooks → Python modules, create one file per backbone:
- `src/models/efficientnet.py` — `EfficientSkinClassifier` + `load_efficientnet_b1(path, device)`
- `src/models/convnext.py` — `ConvNeXtSkinClassifier` + `load_convnext_tiny(path, device)`
- `src/models/swin.py` — `SwinSkinClassifier` + `load_swin_tiny(path, device)`

The load function already exists in each notebook's final cell. Define the class once in the
`.py` file — do not duplicate between training and load cells as the notebooks currently do.

## agreement_score for ensemble.py

The `Predictions` schema requires `agreement_score: float`. Agreed definition:
**fraction of the three models whose top-1 prediction matches the ensemble top-1**.
Values: 0.33 (one agrees), 0.67 (two agree), 1.0 (all agree).
Implement this in `ensemble.py` before wiring up the pipeline.
