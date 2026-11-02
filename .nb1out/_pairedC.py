"""Paired bootstrap of the ensemble against ConvNeXt v2.

NB1 section 9b hardcodes the comparison against Swin, which the retrain made
obsolete: ConvNeXt v2 is now the best single backbone. Same protocol as NB1 --
2000 resamples, seed 42, clean 565 -- with accuracy, macro-AUC and MCC
implemented directly so no sklearn is needed.
"""
import csv
import sys
from pathlib import PurePosixPath as P

import numpy as np

K = 5


def acc(p, y):
    return (p.argmax(1) == y).mean()


def _auc_binary(score, pos):
    """ROC AUC via the rank/Mann-Whitney identity, with ties averaged."""
    n_pos = pos.sum()
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mauc(p, y):
    vals = [_auc_binary(p[:, k], y == k) for k in range(K)]
    return float(np.mean(vals))


def mcc(p, y):
    """Multiclass MCC from the confusion matrix (Gorodkin's R_K)."""
    pred = p.argmax(1)
    C = np.zeros((K, K), dtype=float)
    np.add.at(C, (y, pred), 1)
    t = C.sum(1)          # true per class
    q = C.sum(0)          # predicted per class
    c = np.trace(C)
    s = C.sum()
    num = c * s - t @ q
    den = np.sqrt(s * s - q @ q) * np.sqrt(s * s - t @ t)
    return float(num / den) if den else 0.0


d, man = sys.argv[1], sys.argv[2]
Z = {k: np.load(f"{d}/cache_{k}_test.npz", allow_pickle=True)
     for k in ["efficientnet_b1", "convnext_tiny", "swin_tiny"]}
labels_full = Z["efficientnet_b1"]["labels"]
paths_full = Z["efficientnet_b1"]["paths"]

keep = {f"{P(r['path']).parent.name}/{P(r['path']).name}"
        for r in csv.DictReader(open(man, encoding="utf-8-sig"))}
keys = [f"{P(str(p)).parent.name}/{P(str(p)).name}" for p in paths_full]
mask = np.array([k in keep for k in keys])
print(f"clean mask: {mask.sum()} of {len(mask)}  (expect 565 of 727)")

y = labels_full[mask]
PRB = {k: Z[k]["probs"][mask] for k in Z}
E, C_, S = PRB["efficientnet_b1"], PRB["convnext_tiny"], PRB["swin_tiny"]
ECS, EC, CS = (E + C_ + S) / 3, (E + C_) / 2, (C_ + S) / 2

print("\nreconstructed, clean 565 (compare with ablation_table.csv):")
for n, p in [("E", E), ("C", C_), ("S", S), ("EC", EC), ("CS", CS), ("ECS", ECS)]:
    print(f"  {n:4s} acc={acc(p,y)*100:8.4f}%  auc={mauc(p,y):.6f}  mcc={mcc(p,y):.6f}")

rng = np.random.default_rng(42)
idx = rng.integers(0, len(y), size=(2000, len(y)))


def paired(a, b, label):
    print(f"\n=== {label} ===")
    for mname, fn in [("accuracy (pp)", lambda p, yy: acc(p, yy) * 100),
                      ("macro-AUC", mauc), ("MCC", mcc)]:
        obs = fn(a, y) - fn(b, y)
        draws, skipped = [], 0
        for row in idx:
            yy = y[row]
            if mname == "macro-AUC" and len(np.unique(yy)) < K:
                skipped += 1
                continue
            draws.append(fn(a[row], yy) - fn(b[row], yy))
        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  {mname:14s} obs={obs:+.6f}  CI=[{lo:+.6f}, {hi:+.6f}]  "
              f"favours_first={(draws>0).mean():.4f}  "
              f"excludes_zero={not (lo <= 0 <= hi)}"
              + (f"  (skipped {skipped})" if skipped else ""))


paired(ECS, C_, "ECS (full ensemble) - C (ConvNeXt v2)")
paired(EC, C_, "EC - C (ConvNeXt v2)")
paired(CS, C_, "CS - C (ConvNeXt v2)")
paired(EC, ECS, "EC - ECS")
