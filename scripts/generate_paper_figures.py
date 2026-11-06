"""Generate the three paper figures from cached .nb1out data.

Fig 3: Model comparison bar chart (accuracy, macro-AUC, MCC)
Fig 4: ROC curves (requires running ensemble inference on the dedup test set)
Fig 5: Confusion matrix + per-class sensitivity

Figs 3 and 5 use only cached CSVs; fig 4 needs the checkpoints and test images.
Run with:  ./venv/Scripts/python.exe scripts/generate_paper_figures.py [--skip-roc]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NB1 = ROOT / ".nb1out"
OUT  = ROOT / "paper"

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
CLASS_SHORT = ["BCC", "Eczema", "Keratosis", "Melanoma", "Tinea"]

# ── palette ──────────────────────────────────────────────────────────────
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CM_CMAP = "Blues"


def load_ablation():
    rows = []
    with open(NB1 / "ablation_table.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_per_class():
    rows = []
    with open(NB1 / "per_class_metrics.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_confusion(config="ECS"):
    with open(NB1 / f"confusion_{config}_counts.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        mat = []
        labels = []
        for row in reader:
            labels.append(row[0])
            mat.append([int(x) for x in row[1:]])
    return np.array(mat), labels


# ── Fig 3: model comparison bar chart ────────────────────────────────────

def fig3_model_comparison(ablation_rows):
    configs_order = ["E", "S", "C", "ECS"]
    config_labels = {
        "E": "EfficientNet-B1",
        "S": "Swin-Tiny",
        "C": "ConvNeXt-Tiny",
        "ECS": "Ensemble (E+C+S)",
    }
    metrics = ["accuracy", "macro_auc", "mcc"]
    metric_labels = ["Accuracy", "Macro-AUC", "MCC"]

    lookup = {r["config"]: r for r in ablation_rows}

    x = np.arange(len(configs_order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [float(lookup[c][m]) for c in configs_order]
        bars = ax.bar(x + i * width, vals, width, label=ml, color=BAR_COLORS[i],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}" if m != "accuracy" else f"{v*100:.2f}%",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([config_labels[c] for c in configs_order], fontsize=9)
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Test Accuracy, Macro-Average AUC, and MCC\n"
                 "(de-duplicated 565-image test set, v2 checkpoints)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUT / "fig3_model_comparison.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Fig 5: confusion matrix ─────────────────────────────────────────────

def fig5_confusion_matrix():
    mat, labels = load_confusion("ECS")
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = mat / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.1, 0.9]})

    # left: heatmap
    ax = axes[0]
    im = ax.imshow(norm, cmap=CM_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASS_SHORT)))
    ax.set_yticks(range(len(CLASS_SHORT)))
    ax.set_xticklabels(CLASS_SHORT, fontsize=9, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_SHORT, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=11)

    for i in range(len(CLASS_SHORT)):
        for j in range(len(CLASS_SHORT)):
            count = mat[i, j]
            pct = norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1%})", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if i == j else "normal")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

    # right: per-class sensitivity bar
    ax2 = axes[1]
    sensitivities = np.diag(norm)
    supports = row_sums.flatten()
    colors_bar = ["#4C72B0"] * len(CLASS_SHORT)
    bars = ax2.barh(range(len(CLASS_SHORT)), sensitivities, color=colors_bar,
                    edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(CLASS_SHORT)))
    ax2.set_yticklabels(CLASS_SHORT, fontsize=9)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Sensitivity (Recall)", fontsize=10)
    ax2.set_title("Per-Class Sensitivity", fontsize=11)
    ax2.invert_yaxis()

    for i, (s, n) in enumerate(zip(sensitivities, supports)):
        ax2.text(s + 0.01, i, f"{s:.4f}  ({int(np.diag(mat)[i])}/{int(n)})",
                 va="center", fontsize=9)

    ax2.axvline(0.95, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.text(0.952, len(CLASS_SHORT) - 0.5, "95%", color="red", alpha=0.5, fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Ensemble (EfficientNet-B1 + ConvNeXt-Tiny + Swin-Tiny)\n"
                 "De-duplicated 565-image test set, v2 checkpoints",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    path = OUT / "fig5_confusion_matrix.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Fig 4: ROC curves (requires inference) ───────────────────────────────

def fig4_roc_curves():
    """Run ensemble inference on the dedup test set and plot ROC curves."""
    import torch
    from sklearn.metrics import roc_curve, auc

    sys.path.insert(0, str(ROOT))
    from src.models.ensemble import SkinEnsemble
    from src.utils.config import get_checkpoint_paths
    from src.preprocessing.inference_transform import preprocess_image
    from PIL import Image

    print("  Loading ensemble checkpoints...")
    ckpt = get_checkpoint_paths()
    ensemble = SkinEnsemble(ckpt)

    # Find the test directory: data/raw/IPD_Datasets_Final_Split/test/<Class>/
    test_candidates = [
        ROOT / "data" / "raw" / "IPD_Datasets_Final_Split" / "test",
        ROOT / "data" / "raw" / "test",
    ]
    test_dir = None
    for d in test_candidates:
        if d.is_dir():
            test_dir = d
            break
    if test_dir is None:
        print("  ERROR: no test directory found")
        return

    # collect test images per class (all 727; no dedup manifest locally)
    all_images = []
    all_labels = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = test_dir / cls_name
        if not cls_dir.is_dir():
            print(f"    WARNING: no directory for {cls_name} at {cls_dir}, skipping")
            continue

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                all_images.append(img_path)
                all_labels.append(cls_idx)

    print(f"  Found {len(all_images)} test images across {len(set(all_labels))} classes")

    # run inference
    all_probs = []
    with torch.no_grad():
        for i, img_path in enumerate(all_images):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    inference {i+1}/{len(all_images)}...")
            img = Image.open(img_path)
            tensor = preprocess_image(img)
            pred = ensemble.predict(tensor)
            probs = [pred.class_probabilities[c] for c in CLASSES]
            all_probs.append(probs)

    probs_array = np.array(all_probs)  # (N, 5)
    labels_array = np.array(all_labels)  # (N,)

    # plot ROC
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52", "#DD8452"]

    for cls_idx, (cls_short, color) in enumerate(zip(CLASS_SHORT, colors)):
        y_true_bin = (labels_array == cls_idx).astype(int)
        y_score = probs_array[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{cls_short} (AUC = {roc_auc:.4f})")

    # macro-average ROC
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    for cls_idx in range(len(CLASSES)):
        y_true_bin = (labels_array == cls_idx).astype(int)
        y_score = probs_array[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(CLASSES)
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.2, linestyle="--",
            label=f"Macro-average (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, lw=0.8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    n_imgs = len(labels_array)
    ax.set_title("ROC Curves — Ensemble (EfficientNet-B1 + ConvNeXt-Tiny + Swin-Tiny)\n"
                 f"{n_imgs}-image test set, v2 checkpoints", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path_pdf = OUT / "fig4_roc.pdf"
    path_jpg = OUT / "fig4_roc.jpg"
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(path_jpg, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path_pdf}")
    print(f"  Saved {path_jpg}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from cached data")
    parser.add_argument("--skip-roc", action="store_true",
                        help="Skip fig4 ROC (requires checkpoint inference)")
    parser.add_argument("--only", choices=["fig3", "fig4", "fig5"],
                        help="Generate only one figure")
    args = parser.parse_args()

    ablation = load_ablation()

    if args.only:
        if args.only == "fig3":
            print("Generating Fig 3 (model comparison)...")
            fig3_model_comparison(ablation)
        elif args.only == "fig4":
            print("Generating Fig 4 (ROC curves)...")
            fig4_roc_curves()
        elif args.only == "fig5":
            print("Generating Fig 5 (confusion matrix)...")
            fig5_confusion_matrix()
        return

    print("Generating Fig 3 (model comparison)...")
    fig3_model_comparison(ablation)

    print("Generating Fig 5 (confusion matrix + per-class sensitivity)...")
    fig5_confusion_matrix()

    if args.skip_roc:
        print("Skipping Fig 4 (ROC) — pass without --skip-roc to generate")
    else:
        print("Generating Fig 4 (ROC curves — running inference, may take 15+ min on CPU)...")
        fig4_roc_curves()

    print("Done.")


if __name__ == "__main__":
    main()
