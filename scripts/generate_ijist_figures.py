"""
Regenerate Figures 3-6 for the IJIST manuscript (paper_ijist/) from the
cached per-image test probabilities, on the de-duplicated 565-image test
set, using only the three proposed-ensemble backbones (E, C, S -> E+C+S).

Why this script exists: the figures shipped in paper/fig{3,4,5}_*.pdf are of
unconfirmed provenance (FACT_CHECK.md E11 -- their filenames match no
notebook's actual output path) and in any case were not regenerated after
the v2 retrain / de-duplication / calibration work. This script rebuilds
them directly from .nb1out/cache_*_test.npz, which both new bootstrap
scripts (cs_vs_convnext_analysis.py, cs_vs_ecs_analysis.py) already
validated as internally consistent (matching path/label ordering across
backbones).

Palette: the dataviz skill's validated default categorical palette (first
five slots: blue/orange/aqua/yellow/magenta), fixed order across all
figures so a class keeps the same color everywhere in the paper. Sequential
blue ramp for magnitude (confusion-matrix heatmap, ROC-curve shading).
Colorblind-safe per the IJIST checklist's figure guidance.

Outputs (300 dpi PNG for review + PDF for submission), written to
paper_ijist/figures/:
  Figure_3.pdf/png  -- model comparison (accuracy / macro-AUC / MCC bars)
  Figure_4.pdf/png  -- ROC curves, per class + macro-average
  Figure_5.pdf/png  -- confusion matrix + per-class sensitivity
  Figure_6.pdf/png  -- calibration: reliability diagram + ECE/MCE bar chart

Run: ./venv/Scripts/python.exe scripts/generate_ijist_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent.parent
NB1OUT = ROOT / ".nb1out"
OUTDIR = ROOT / "paper_ijist" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
CLEAN_MANIFEST = NB1OUT / "val_dedup_results" / "test_clean_manifest.csv"

# ---- dataviz skill: validated default categorical palette, light mode ----
CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
}
CLASS_COLOR = {
    "Basal_Cell_Carcinoma": CAT["blue"],
    "Eczema": CAT["orange"],
    "Keratosis": CAT["aqua"],
    "Melanoma": CAT["yellow"],
    "Tinea": CAT["magenta"],
}
CLASS_LABEL = {
    "Basal_Cell_Carcinoma": "BCC",
    "Eczema": "Eczema",
    "Keratosis": "Keratosis",
    "Melanoma": "Melanoma",
    "Tinea": "Tinea",
}
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_clean_paths() -> set[str]:
    paths = set()
    with open(CLEAN_MANIFEST, encoding="utf-8") as f:
        next(f)
        for line in f:
            paths.add(line.rsplit(",", 1)[0].strip())
    return paths


def load_backbone(name: str, clean_paths: set[str]):
    d = np.load(NB1OUT / f"cache_{name}_test.npz", allow_pickle=True)
    probs, labels, paths, classes = d["probs"], d["labels"], d["paths"], d["classes"]
    mask = np.array([p in clean_paths for p in paths])
    return probs[mask], labels[mask], paths[mask], list(classes)


def macro_auc(probs, labels, n_classes):
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    aucs = []
    for c in range(n_classes):
        if y_bin[:, c].sum() in (0, len(y_bin)):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs))


def mcc(labels, pred):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(labels, pred)


def main() -> None:
    clean_paths = load_clean_paths()
    probs_e, labels, paths_e, classes = load_backbone("efficientnet_b1", clean_paths)
    probs_c, _, paths_c, _ = load_backbone("convnext_tiny", clean_paths)
    probs_s, _, paths_s, _ = load_backbone("swin_tiny", clean_paths)
    assert np.array_equal(paths_e, paths_c) and np.array_equal(paths_c, paths_s)
    n_classes = len(classes)
    n = len(labels)
    print(f"Clean test set: n={n}, classes={classes}")

    probs_ecs = (probs_e + probs_c + probs_s) / 3.0
    configs = {
        "EfficientNet-B1": probs_e,
        "Swin-Tiny": probs_s,
        "ConvNeXt-Tiny": probs_c,
        "E+C+S (proposed)": probs_ecs,
    }

    # ================================================================
    # Figure 3: model comparison (accuracy / macro-AUC / MCC)
    # ================================================================
    names = list(configs.keys())
    accs, aucs_, mccs = [], [], []
    for p in configs.values():
        pred = p.argmax(axis=1)
        accs.append((pred == labels).mean() * 100)
        aucs_.append(macro_auc(p, labels, n_classes))
        mccs.append(mcc(labels, pred))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    bar_colors = [CAT["blue"], CAT["orange"], CAT["aqua"], CAT["yellow"]]
    metrics = [("Accuracy (%)", accs, (0, 100)), ("Macro-AUC", aucs_, (0.9, 1.0)), ("MCC", mccs, (0, 1.0))]
    for ax, (title, vals, ylim) in zip(axes, metrics):
        bars = ax.bar(range(len(names)), vals, color=bar_colors, width=0.6,
                       edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["Eff-B1", "Swin-T", "ConvNeXt-T", "E+C+S"], fontsize=7, rotation=20, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            label = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                     ha="center", va="bottom", fontsize=6.5, color=INK)
    fig.suptitle("Model Comparison: Accuracy, Macro-AUC, and MCC (clean 565-image test set)", fontsize=9, y=1.04)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Figure_3.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Figure_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure_3.pdf/.png")

    # ================================================================
    # Figure 4: ROC curves (per class + macro-average), ensemble only
    # ================================================================
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    all_fpr = np.linspace(0, 1, 200)
    tprs = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs_ecs[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLOR[classes[c]], linewidth=1.4,
                label=f"{CLASS_LABEL[classes[c]]} (AUC={roc_auc:.3f})")
        tprs.append(np.interp(all_fpr, fpr, tpr))
    macro_tpr = np.mean(tprs, axis=0)
    macro_auc_val = auc(all_fpr, macro_tpr)
    ax.plot(all_fpr, macro_tpr, color=INK, linewidth=1.8, linestyle="--",
            label=f"Macro-average (AUC={macro_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: E+C+S Ensemble (clean 565-image test set)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Figure_4.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Figure_4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure_4.pdf/.png")

    # ================================================================
    # Figure 5: confusion matrix + per-class sensitivity
    # ================================================================
    pred_ecs = probs_ecs.argmax(axis=1)
    cm = confusion_matrix(labels, pred_ecs, labels=list(range(n_classes)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    im = ax1.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    short_labels = [CLASS_LABEL[c] for c in classes]
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(short_labels, fontsize=7, rotation=30, ha="right")
    ax1.set_yticklabels(short_labels, fontsize=7)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (row-normalized)", fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val == 0:
                continue
            txt_color = "white" if cm_norm[i, j] > 0.5 else INK
            ax1.text(j, i, str(val), ha="center", va="center", fontsize=6.5, color=txt_color)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    sens = np.diag(cm_norm) * 100
    bars = ax2.barh(range(n_classes), sens, color=[CLASS_COLOR[c] for c in classes], height=0.6)
    ax2.set_yticks(range(n_classes))
    ax2.set_yticklabels(short_labels, fontsize=7)
    ax2.set_xlabel("Sensitivity (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Per-Class Sensitivity", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    for b, v, support in zip(bars, sens, cm.sum(axis=1)):
        ax2.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.2f}% (n={support})",
                  va="center", fontsize=6.5, color=INK)
    fig.suptitle("Confusion Matrix and Per-Class Sensitivity: E+C+S Ensemble", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Figure_5.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Figure_5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure_5.pdf/.png")

    # ================================================================
    # Figure 6: calibration -- reliability diagram + ECE/MCE bar chart
    # ================================================================
    results_path = NB1OUT / "results.json"
    cal = None
    temp = None
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            res = json.load(f)
        cal = res.get("calibration")
        temp = res.get("temperature_scaling")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Reliability diagram: ensemble before vs after temperature scaling
    def conf_acc_bins(probs, labels, n_bins=10):
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        mids, accs_, confs_ = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
            if in_bin.sum() == 0:
                continue
            mids.append((lo + hi) / 2)
            accs_.append(correct[in_bin].mean())
            confs_.append(conf[in_bin].mean())
        return np.array(mids), np.array(accs_), np.array(confs_)

    mids, bin_acc, bin_conf = conf_acc_bins(probs_ecs, labels)
    ax1.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle=":", label="Perfect calibration")
    ax1.plot(bin_conf, bin_acc, marker="o", color=CAT["blue"], linewidth=1.4, markersize=4,
              label="E+C+S (before scaling)")
    if temp is not None:
        T = temp["T"]
        probs_scaled = probs_ecs ** (1.0 / T)
        probs_scaled = probs_scaled / probs_scaled.sum(axis=1, keepdims=True)
        mids2, bin_acc2, bin_conf2 = conf_acc_bins(probs_scaled, labels)
        ax1.plot(bin_conf2, bin_acc2, marker="s", color=CAT["orange"], linewidth=1.4, markersize=4,
                  label=f"After scaling (T={T:.3f})")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Reliability Diagram", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=6.5, loc="upper left", frameon=False)

    # ECE bar chart across configs
    if cal is not None:
        order = ["E", "S", "C", "ECS"]
        disp_names = ["Eff-B1", "Swin-T", "ConvNeXt-T", "E+C+S"]
        eces = [cal[k]["ece"] for k in order]
        overconf = [cal[k]["overconfidence"] for k in order]
        bars = ax2.bar(range(len(order)), eces, color=bar_colors, width=0.6, edgecolor="white", linewidth=0.5)
        ax2.set_xticks(range(len(order)))
        ax2.set_xticklabels(disp_names, fontsize=7, rotation=20, ha="right")
        ax2.set_ylabel("ECE (10 bins)")
        ax2.set_title("Expected Calibration Error", fontsize=9)
        ax2.set_ylim(0, max(eces) * 1.35)  # headroom so two-line bar labels clear the title
        ax2.spines[["top", "right"]].set_visible(False)
        for b, v, oc in zip(bars, eces, overconf):
            sign = "+" if oc >= 0 else "−"
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}\n({sign}{abs(oc):.4f})",
                      ha="center", va="bottom", fontsize=6, color=INK)
    fig.suptitle("Calibration: Reliability and Expected Calibration Error", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Figure_6.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "Figure_6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure_6.pdf/.png")

    print(f"\nAll figures written to {OUTDIR}")


if __name__ == "__main__":
    main()
