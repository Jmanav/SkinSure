"""Analyse `.nb1out/xai_aopc_raw.csv`: Grad-CAM vs random baselines on the deletion/AOPC test.

Reads the cached per-image rows written by `scripts/xai_aopc_experiment.py` -- the models
are never reloaded, so the analysis is cheap to re-run.

Pairing: AOPC is a per-image measurement, so Grad-CAM and each baseline are compared with
a PAIRED test on the same images. Each random baseline is averaged over its seeds first,
giving one number per (image, method) before pairing. Wilcoxon signed-rank is the headline
test; a sign test is reported alongside as a distribution-free check.

scipy is not installed in this venv, so Wilcoxon is implemented here: exact null for
n <= 20 by enumeration, otherwise the normal approximation with tie and continuity
correction. Verified against the textbook worked example in `_selftest()`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / ".nb1out" / "xai_aopc_raw.csv"
CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
BASELINES = ["random7x7", "randompixel"]


# --------------------------------------------------------------------------- statistics

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (same convention scipy.stats.rankdata uses)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon(diffs: np.ndarray) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank test. Returns (W, p, n_nonzero).

    Zero differences are dropped (pratt/wilcox "zsplit"-free convention, same as
    scipy's default `zero_method='wilcox'`). W is the smaller of the two signed
    rank sums.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    ranks = _rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    w = min(w_plus, w_minus)

    has_ties = len(np.unique(np.abs(d))) != n
    if n <= 20 and not has_ties:
        # Exact null: every sign assignment is equally likely.
        total = 0
        count = 0
        for signs in product((0, 1), repeat=n):
            s = sum(r for r, keep in zip(ranks, signs) if keep)
            total += 1
            if min(s, ranks.sum() - s) <= w:
                count += 1
        return float(w), min(1.0, count / total), n

    mean_w = n * (n + 1) / 4.0
    # tie correction on |d|
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = sum(c**3 - c for c in counts)
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return float(w), float("nan"), n
    z = (abs(w - mean_w) - 0.5) / math.sqrt(var_w)   # continuity correction
    p = 2.0 * (1.0 - _norm_cdf(z))
    return float(w), min(1.0, max(0.0, p)), n


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def sign_test(diffs: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided sign test. Returns (n_positive, n_nonzero, p)."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    pos = int((d > 0).sum())
    if n == 0:
        return 0, 0, float("nan")
    k = min(pos, n - pos)
    tail = sum(math.exp(_log_binom(n, i) - n * math.log(2.0)) for i in range(k + 1))
    return pos, n, min(1.0, 2.0 * tail)


def boot_ci(values: np.ndarray, seed: int = 7, reps: int = 20000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _selftest() -> None:
    """Check the exact Wilcoxon and sign test against hand-computable answers.

    Case 1: d = 1,-2,3,4,5,6,-7,8,9,10. |d| has no ties, so ranks equal |d|; the
    negative rank sum is 2+7 = 9, which is the smaller side, so W = 9.
    Case 2: nine all-positive differences. Every rank is positive, so W = 0, and the
    exact two-sided p is 2/2^9 = 0.00390625 -- the same value scipy.stats.wilcoxon
    returns for this input. The sign test must agree exactly here.
    """
    w, p, n = wilcoxon(np.array([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, -7.0, 8.0, 9.0, 10.0]))
    assert (n, w) == (10, 9.0), (n, w)
    assert abs(p - 0.064453125) < 1e-9, p

    d2 = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    w2, p2, n2 = wilcoxon(d2)
    assert (n2, w2) == (9, 0.0), (n2, w2)
    assert abs(p2 - 2 / 2**9) < 1e-12, p2
    assert abs(sign_test(d2)[2] - 2 / 2**9) < 1e-12

    assert np.allclose(_rankdata(np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])),
                       [1.5, 1.5, 3.0, 5.0, 5.0, 5.0])
    print("selftest ok: exact Wilcoxon, sign test and tied ranks all match known values")


# --------------------------------------------------------------------------- loading

def load(raw: Path):
    """-> (per_image, meta): per_image[path][method] = mean AOPC over that method's seeds."""
    per_image: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict[str, str]] = {}
    curves: dict[str, list[list[float]]] = defaultdict(list)
    pcts: list[int] = []
    with raw.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        pcts = [int(f[1:]) for f in reader.fieldnames if f.startswith("p") and f[1:].isdigit()]
        for rec in reader:
            per_image[rec["path"]][rec["method"]].append(float(rec["aopc"]))
            meta[rec["path"]] = {
                "true_class": rec["true_class"],
                "pred_class": rec["pred_class"],
                "correct": rec["correct"],
                "pred_conf": rec["pred_conf"],
                "secs": rec["secs"] if rec["method"] == "gradcam" else meta.get(
                    rec["path"], {}).get("secs", ""),
            }
            curves[rec["method"]].append([float(rec[f"p{p}"]) for p in pcts])
    collapsed = {
        path: {m: float(np.mean(v)) for m, v in methods.items()}
        for path, methods in per_image.items()
    }
    return collapsed, meta, curves, pcts


def fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.4f}"


def pfmt(p: float) -> str:
    if p != p:
        return "nan"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


# --------------------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--md", type=Path, default=None,
                    help="also write the tables to this markdown file")
    args = ap.parse_args()

    if args.selftest:
        _selftest()

    per_image, meta, curves, pcts = load(args.raw)
    paths = sorted(per_image)
    n = len(paths)
    methods = ["gradcam"] + [b for b in BASELINES if any(b in per_image[p] for p in paths)]

    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    per_cls = Counter(meta[p]["true_class"] for p in paths)
    spread = ", ".join(f"{c} {per_cls[c]}" for c in CLASSES if per_cls[c])
    say(f"n = {n} images from the de-duplicated 565-image clean test set ({spread})")
    if len(per_cls) < len(CLASSES):
        say()
        say(f"**PARTIAL RUN -- only {len(per_cls)} of {len(CLASSES)} classes present. "
            f"Not a result: the sample is ordered by class, so an incomplete run is "
            f"class-biased, not a random subset.**")
    say()

    # ---- headline
    say("## Overall AOPC (higher = heatmap found pixels the prediction depends on)")
    say()
    say("| method | mean AOPC | 95% CI (bootstrap) | median |")
    say("|---|---|---|---|")
    vals = {m: np.array([per_image[p][m] for p in paths]) for m in methods}
    for m in methods:
        lo, hi = boot_ci(vals[m])
        say(f"| {m} | {fmt(vals[m].mean())} | [{fmt(lo)}, {fmt(hi)}] | "
            f"{fmt(float(np.median(vals[m])))} |")
    say()

    say("## Paired comparisons (Grad-CAM minus baseline, same images)")
    say()
    say("| comparison | mean diff | 95% CI | Wilcoxon W | p | sign test | p |")
    say("|---|---|---|---|---|---|---|")
    for b in methods[1:]:
        d = vals["gradcam"] - vals[b]
        lo, hi = boot_ci(d)
        w, p, nz = wilcoxon(d)
        pos, nn, sp = sign_test(d)
        say(f"| gradcam - {b} | {fmt(d.mean())} | [{fmt(lo)}, {fmt(hi)}] | "
            f"{fmt(w)} (n={nz}) | {pfmt(p)} | {pos}/{nn} positive | {pfmt(sp)} |")
    say()
    say("**`random7x7` is the fair comparator; `randompixel` is a sanity check, not a "
        "rival.** Both baselines delete the same NUMBER of pixels, but not the same KIND "
        "of region. `random7x7` draws noise on the 7x7 Grad-CAM grid and upsamples it, so "
        "its deletions are contiguous blobs matched in spatial structure to a Grad-CAM "
        "map -- the only difference is whether the blobs are in informative places. "
        "`randompixel` scatters i.i.d. noise at full 224x224 resolution, so it removes a "
        "fine-grained speckle that shreds local texture everywhere at once and pushes the "
        "image far off the training distribution. That collapses p(class) no matter what "
        "it covers, which is why it posts the highest AOPC while explaining nothing. A "
        "high `randompixel` AOPC therefore measures the models' fragility to high-frequency "
        "corruption, NOT localisation quality; read `gradcam - random7x7` as the result and "
        "treat `gradcam - randompixel` as confirmation that AOPC is sensitive to "
        "out-of-distribution damage and so must be compared only against a "
        "structure-matched baseline.")
    say()

    # ---- per class
    say("## Per-class AOPC (per-class n is in the table; intervals are wide and each "
        "row is individually underpowered)")
    say()
    header = "| true class | n | " + " | ".join(methods) + " | gradcam-random7x7 | Wilcoxon p |"
    say(header)
    say("|---" * (len(methods) + 4) + "|")
    for cls in CLASSES:
        sel = [p for p in paths if meta[p]["true_class"] == cls]
        if not sel:
            continue
        cells = []
        for m in methods:
            v = np.array([per_image[p][m] for p in sel])
            lo, hi = boot_ci(v)
            cells.append(f"{fmt(v.mean())} [{fmt(lo)},{fmt(hi)}]")
        d = np.array([per_image[p]["gradcam"] - per_image[p]["random7x7"] for p in sel])
        _, p_cls, _ = wilcoxon(d)
        say(f"| {cls} | {len(sel)} | " + " | ".join(cells)
            + f" | {fmt(d.mean())} | {pfmt(p_cls)} |")
    say()

    # ---- correct vs incorrect
    say("## Split by whether the ensemble's prediction was correct")
    say()
    say("| subset | n | " + " | ".join(methods) + " | gradcam-random7x7 | Wilcoxon p |")
    say("|---" * (len(methods) + 4) + "|")
    for label, want in (("correct", "1"), ("incorrect", "0")):
        sel = [p for p in paths if meta[p]["correct"] == want]
        if not sel:
            continue
        cells = [fmt(float(np.mean([per_image[p][m] for p in sel]))) for m in methods]
        d = np.array([per_image[p]["gradcam"] - per_image[p]["random7x7"] for p in sel])
        _, p_sub, _ = wilcoxon(d)
        say(f"| {label} | {len(sel)} | " + " | ".join(cells)
            + f" | {fmt(d.mean())} | {pfmt(p_sub)} |")
    say()

    # ---- deletion curves
    say("## Mean deletion curve: p(predicted class) as the top-k pixels are blanked")
    say()
    say("| deleted % | " + " | ".join(methods) + " |")
    say("|---" * (len(methods) + 1) + "|")
    curve_means = {m: np.array(curves[m]).mean(axis=0) for m in methods}
    for i, pct in enumerate(pcts):
        say(f"| {pct}% | " + " | ".join(fmt(curve_means[m][i]) for m in methods) + " |")
    say()

    # ---- win rate / effect size
    say("## Effect size and win rate (vs random7x7)")
    say()
    d = vals["gradcam"] - vals["random7x7"]
    wins = int((d > 0).sum())
    sd = float(d.std(ddof=1))
    say(f"- Grad-CAM scored higher on **{wins}/{n}** images ({100*wins/n:.1f}%).")
    say(f"- Mean paired difference {d.mean():.4f}, SD {sd:.4f}, "
        f"Cohen's dz = {d.mean()/sd:.3f}." if sd > 0 else "- zero variance")
    rbc = ((d > 0).sum() - (d < 0).sum()) / max(1, (d != 0).sum())
    say(f"- Matched-pairs rank-biserial correlation {rbc:+.3f}.")
    say()

    gc_secs = np.array([float(meta[p]["secs"]) for p in paths if meta[p]["secs"]])
    if len(gc_secs):
        say(f"Grad-CAM build + deletion curve: mean {gc_secs.mean():.2f} s/image, "
            f"median {np.median(gc_secs):.2f}, max {gc_secs.max():.2f}.")

    if args.md:
        args.md.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nwrote {args.md}")
    return out


if __name__ == "__main__":
    main()
