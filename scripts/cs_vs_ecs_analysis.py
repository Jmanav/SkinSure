"""
C+S (ConvNeXt + Swin) vs. E+C+S (the old three-backbone proposal) -- the
paired-bootstrap comparison called out as "[PENDING: insert result]" in the
IJIST draft's ablation discussion (main_updated.tex Sec. 6.2).

Uses the already-cached per-image test probabilities (.nb1out/cache_*_test.npz,
written by paper/experiments/01_eval_and_ablation.ipynb) -- no Kaggle run needed.
Same method, seed and clean-565 filtering as scripts/cs_vs_convnext_analysis.py,
so the two results are directly comparable.

Run: ./venv/Scripts/python.exe scripts/cs_vs_ecs_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
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

    probs_e, labels_e, paths_e, classes = load_backbone("efficientnet_b1", clean_paths)
    probs_c, labels_c, paths_c, classes_c = load_backbone("convnext_tiny", clean_paths)
    probs_s, labels_s, paths_s, classes_s = load_backbone("swin_tiny", clean_paths)
    assert classes == classes_c == classes_s
    assert np.array_equal(paths_e, paths_c) and np.array_equal(paths_c, paths_s), \
        "cache ordering mismatch across backbones -- align by path first"
    assert np.array_equal(labels_e, labels_c) and np.array_equal(labels_c, labels_s)

    labels = labels_e
    n_classes = len(classes)
    n = len(labels)
    print(f"Filtered to clean test set: n={n} (expect 565)")

    probs_cs = 0.5 * probs_c + 0.5 * probs_s
    probs_ecs = (probs_e + probs_c + probs_s) / 3.0

    def summarize(probs):
        pred = probs.argmax(axis=1)
        return {
            "accuracy": float((pred == labels).mean()),
            "macro_auc": macro_auc(probs, labels, n_classes),
            "mcc": float(matthews_corrcoef(labels, pred)),
            "ece": ece(probs, labels),
        }

    cs_summary = summarize(probs_cs)
    ecs_summary = summarize(probs_ecs)
    print(f"\nC+S:   acc={cs_summary['accuracy']*100:.2f}%  macro-AUC={cs_summary['macro_auc']:.4f}  "
          f"MCC={cs_summary['mcc']:.4f}  ECE={cs_summary['ece']:.4f}")
    print(f"E+C+S: acc={ecs_summary['accuracy']*100:.2f}%  macro-AUC={ecs_summary['macro_auc']:.4f}  "
          f"MCC={ecs_summary['mcc']:.4f}  ECE={ecs_summary['ece']:.4f}")

    rng = np.random.default_rng(SEED)
    deltas = {"accuracy (pp)": [], "macro-AUC": [], "MCC": [], "ECE": []}
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        l_b = labels[idx]
        p_cs_b = probs_cs[idx]
        p_ecs_b = probs_ecs[idx]

        pred_cs_b = p_cs_b.argmax(axis=1)
        pred_ecs_b = p_ecs_b.argmax(axis=1)

        acc_cs_b = (pred_cs_b == l_b).mean()
        acc_ecs_b = (pred_ecs_b == l_b).mean()
        deltas["accuracy (pp)"].append((acc_cs_b - acc_ecs_b) * 100)

        deltas["macro-AUC"].append(macro_auc(p_cs_b, l_b, n_classes) - macro_auc(p_ecs_b, l_b, n_classes))
        deltas["MCC"].append(matthews_corrcoef(l_b, pred_cs_b) - matthews_corrcoef(l_b, pred_ecs_b))
        deltas["ECE"].append(ece(p_cs_b, l_b) - ece(p_ecs_b, l_b))

    print("\n" + "=" * 70)
    print("Paired bootstrap: C+S minus E+C+S (2000 resamples, seed=42)")
    print("=" * 70)
    observed = {
        "accuracy (pp)": (cs_summary["accuracy"] - ecs_summary["accuracy"]) * 100,
        "macro-AUC": cs_summary["macro_auc"] - ecs_summary["macro_auc"],
        "MCC": cs_summary["mcc"] - ecs_summary["mcc"],
        "ECE": cs_summary["ece"] - ecs_summary["ece"],
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
        print(f"  observed delta (C+S - E+C+S): {obs:+.4f}")
        print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  includes zero: {ci_lo <= 0 <= ci_hi}")
        print(f"  fraction of resamples favouring C+S: {frac_favouring_cs:.4f}")

    out = {
        "n": int(n),
        "cs_summary": cs_summary,
        "ecs_summary": ecs_summary,
        "paired_bootstrap_cs_minus_ecs": bootstrap_results,
    }
    out_path = NB1OUT / "cs_vs_ecs_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
