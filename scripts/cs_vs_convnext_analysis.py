"""
C+S (ConvNeXt + Swin) analysis for the "C+S is the proposed ensemble" paper reframe.

Uses the already-cached per-image test probabilities (.nb1out/cache_*_test.npz,
written by paper/experiments/01_eval_and_ablation.ipynb) -- no Kaggle run needed.

Produces, on the clean 565-image de-duplicated test set:
  1. C+S per-class recall (sensitivity) + 5x5 confusion matrix.
  2. Paired bootstrap CIs for C+S vs. ConvNeXt-Tiny alone: accuracy, macro-AUC,
     MCC, and ECE deltas (2000 resamples, paired by image, same seed=42 as the
     existing ECS-vs-Swin bootstrap in results.json for consistency).

Run: ./venv/Scripts/python.exe scripts/cs_vs_convnext_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent.parent
NB1OUT = ROOT / ".nb1out"
CLEAN_MANIFEST = NB1OUT / "val_dedup_results" / "test_clean_manifest.csv"
SEED = 42
N_RESAMPLES = 2000


def load_clean_paths() -> set[str]:
    paths = set()
    with open(CLEAN_MANIFEST, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            path = line.rsplit(",", 1)[0].strip()
            paths.add(path)
    return paths


def load_backbone(name: str, clean_paths: set[str]):
    d = np.load(NB1OUT / f"cache_{name}_test.npz", allow_pickle=True)
    probs, labels, paths, classes = d["probs"], d["labels"], d["paths"], d["classes"]
    mask = np.array([p in clean_paths for p in paths])
    return probs[mask], labels[mask], paths[mask], list(classes)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total_err = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        total_err += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(total_err)


def macro_auc(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    aucs = []
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0 or y_bin[:, c].sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin[:, c], probs[:, c]))
    return float(np.mean(aucs))


def main() -> None:
    clean_paths = load_clean_paths()
    print(f"Clean manifest: {len(clean_paths)} paths")

    probs_c, labels_c, paths_c, classes = load_backbone("convnext_tiny", clean_paths)
    probs_s, labels_s, paths_s, classes_s = load_backbone("swin_tiny", clean_paths)
    assert classes == classes_s
    assert np.array_equal(paths_c, paths_s), "ConvNeXt/Swin cache ordering mismatch -- align by path first"
    assert np.array_equal(labels_c, labels_s)

    labels = labels_c
    n_classes = len(classes)
    n = len(labels)
    print(f"Filtered to clean test set: n={n} (expect 565)")

    probs_cs = 0.5 * probs_c + 0.5 * probs_s  # equal-weight 2-model average

    # ---------------------------------------------------------- 1. C+S confusion matrix / recall --
    pred_cs = probs_cs.argmax(axis=1)
    cm = confusion_matrix(labels, pred_cs, labels=list(range(n_classes)))

    print("\n" + "=" * 70)
    print("C+S (ConvNeXt + Swin) confusion matrix, clean 565-image test set")
    print("rows = true, cols = predicted")
    print("=" * 70)
    header = " " * 24 + "".join(f"{c[:10]:>12}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        print(f"{c:<24}" + "".join(f"{cm[i,j]:>12d}" for j in range(n_classes)))

    print("\nPer-class recall (sensitivity):")
    recalls = {}
    for i, c in enumerate(classes):
        support = cm[i].sum()
        recall = cm[i, i] / support if support else float("nan")
        recalls[c] = {"recall": float(recall), "support": int(support)}
        print(f"  {c:<24} {recall*100:6.2f}%  (n={support})")

    acc_cs = (pred_cs == labels).mean()
    mcc_cs = matthews_corrcoef(labels, pred_cs)
    auc_cs = macro_auc(probs_cs, labels, n_classes)
    ece_cs = ece(probs_cs, labels)
    print(f"\nC+S summary: acc={acc_cs*100:.2f}%  macro-AUC={auc_cs:.4f}  MCC={mcc_cs:.4f}  ECE={ece_cs:.4f}")

    # Eczema <-> Tinea bidirectional confusion (Mandate 1, Section 6.3)
    ecz_i, tin_i = classes.index("Eczema"), classes.index("Tinea")
    ecz_to_tin = cm[ecz_i, tin_i] / cm[ecz_i].sum() * 100
    tin_to_ecz = cm[tin_i, ecz_i] / cm[tin_i].sum() * 100
    print(f"\nEczema->Tinea misclassification: {ecz_to_tin:.1f}%")
    print(f"Tinea->Eczema misclassification: {tin_to_ecz:.1f}%")

    # ------------------------------------------------- 2. paired bootstrap: C+S vs ConvNeXt alone --
    pred_c = probs_c.argmax(axis=1)
    acc_c = (pred_c == labels).mean()
    mcc_c = matthews_corrcoef(labels, pred_c)
    auc_c = macro_auc(probs_c, labels, n_classes)
    ece_c = ece(probs_c, labels)

    rng = np.random.default_rng(SEED)
    deltas = {"accuracy (pp)": [], "macro-AUC": [], "MCC": [], "ECE": []}
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        l_b = labels[idx]
        p_cs_b = probs_cs[idx]
        p_c_b = probs_c[idx]

        pred_cs_b = p_cs_b.argmax(axis=1)
        pred_c_b = p_c_b.argmax(axis=1)

        acc_cs_b = (pred_cs_b == l_b).mean()
        acc_c_b = (pred_c_b == l_b).mean()
        deltas["accuracy (pp)"].append((acc_cs_b - acc_c_b) * 100)

        deltas["macro-AUC"].append(macro_auc(p_cs_b, l_b, n_classes) - macro_auc(p_c_b, l_b, n_classes))
        deltas["MCC"].append(matthews_corrcoef(l_b, pred_cs_b) - matthews_corrcoef(l_b, pred_c_b))
        deltas["ECE"].append(ece(p_cs_b, l_b) - ece(p_c_b, l_b))

    print("\n" + "=" * 70)
    print("Paired bootstrap: C+S minus ConvNeXt-Tiny alone (2000 resamples, seed=42)")
    print("=" * 70)
    observed = {
        "accuracy (pp)": (acc_cs - acc_c) * 100,
        "macro-AUC": auc_cs - auc_c,
        "MCC": mcc_cs - mcc_c,
        "ECE": ece_cs - ece_c,
    }
    bootstrap_results = {}
    for metric, obs in observed.items():
        arr = np.array(deltas[metric])
        ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
        frac_favouring_cs = float((arr > 0).mean()) if metric != "ECE" else float((arr < 0).mean())
        bootstrap_results[metric] = {
            "observed": float(obs),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "mean": float(arr.mean()),
            "frac_favouring_cs": frac_favouring_cs,
            "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
            "n": N_RESAMPLES,
        }
        direction = "lower is better (ECE)" if metric == "ECE" else "higher is better"
        print(f"\n{metric} ({direction}):")
        print(f"  observed delta (C+S - ConvNeXt): {obs:+.4f}")
        print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  includes zero: {ci_lo <= 0 <= ci_hi}")
        print(f"  fraction of resamples favouring C+S: {frac_favouring_cs:.4f}")

    out = {
        "n": int(n),
        "cs_summary": {"accuracy": float(acc_cs), "macro_auc": float(auc_cs), "mcc": float(mcc_cs), "ece": float(ece_cs)},
        "convnext_summary": {"accuracy": float(acc_c), "macro_auc": float(auc_c), "mcc": float(mcc_c), "ece": float(ece_c)},
        "cs_per_class_recall": recalls,
        "cs_confusion_matrix": cm.tolist(),
        "classes": classes,
        "eczema_to_tinea_pct": float(ecz_to_tin),
        "tinea_to_eczema_pct": float(tin_to_ecz),
        "paired_bootstrap_cs_minus_convnext": bootstrap_results,
    }
    out_path = NB1OUT / "cs_vs_convnext_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
