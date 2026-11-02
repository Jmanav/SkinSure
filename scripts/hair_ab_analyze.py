"""Analysis of the paired hair-removal A/B experiment.

Reads `.nb1out/hair_ab_raw.csv` (written by `hair_ab_experiment.py`) and reports
firing rate, whole-set and fired-subset metrics, McNemar's test and Wilson CIs.
Pure stdlib + numpy (scipy is not installed in this venv), so every statistic is
computed from first principles.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
RAW = ROOT / ".nb1out" / "hair_ab_raw.csv"


# ---------- statistics, from scratch ----------

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_two_sided_exact(b: int, n: int) -> float:
    """Exact two-sided binomial test of b successes in n trials against p=0.5."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pmf[b]
    return min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-12)))


def chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 df = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Paired McNemar on correct/incorrect outcomes.

    b = A right, B wrong (hair removal broke it)
    c = A wrong, B right (hair removal fixed it)
    """
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    out = {
        "both_correct": int(np.sum(a_correct & b_correct)),
        "both_wrong": int(np.sum(~a_correct & ~b_correct)),
        "b_A_right_B_wrong": b,
        "c_A_wrong_B_right": c,
        "n_discordant": n,
        "p_exact": binom_two_sided_exact(b, n),
    }
    if n > 0:
        chi_cc = (abs(b - c) - 1) ** 2 / n if abs(b - c) >= 1 else 0.0
        out["chi2_cc"] = chi_cc
        out["p_chi2_cc"] = chi2_sf_1df(chi_cc)
    else:
        out["chi2_cc"] = float("nan")
        out["p_chi2_cc"] = 1.0
    return out


def auc_binary(scores: np.ndarray, pos: np.ndarray) -> float:
    """ROC AUC via the rank (Mann-Whitney U) identity, with tie-aware midranks."""
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=np.float64)
    r[order] = ranks
    return (r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def macro_auc(probs: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {c: auc_binary(probs[:, k], y == k) for k, c in enumerate(CLASSES)}
    vals = [v for v in per.values() if not math.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


# ---------- reporting ----------

def block(title: str, y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> None:
    n = len(y)
    print(f"\n{'=' * 78}\n{title}  (n={n})\n{'=' * 78}")
    if n == 0:
        print("  empty subset")
        return

    preda, predb = pa.argmax(1), pb.argmax(1)
    ca, cb = preda == y, predb == y

    acc_a, acc_b = ca.mean(), cb.mean()
    la, ha = wilson(int(ca.sum()), n)
    lb, hb = wilson(int(cb.sum()), n)
    print(f"\naccuracy   A(baseline) {acc_a:7.4f}  [{la:.4f}, {ha:.4f}]  {int(ca.sum())}/{n}")
    print(f"           B(hair rm)  {acc_b:7.4f}  [{lb:.4f}, {hb:.4f}]  {int(cb.sum())}/{n}")
    print(f"           delta B-A   {acc_b - acc_a:+7.4f}  ({100 * (acc_b - acc_a):+.2f} pp)")

    ma, pera = macro_auc(pa, y)
    mb, perb = macro_auc(pb, y)
    print(f"\nmacro AUC (one-vs-rest)   A {ma:.4f}   B {mb:.4f}   delta {mb - ma:+.4f}")
    print(f"  {'class':<22}{'AUC A':>9}{'AUC B':>9}{'delta':>9}")
    for c in CLASSES:
        print(f"  {c:<22}{pera[c]:9.4f}{perb[c]:9.4f}{perb[c] - pera[c]:+9.4f}")

    print(f"\nper-class sensitivity / recall")
    print(f"  {'class':<22}{'n':>5}{'sens A':>18}{'sens B':>18}{'delta':>9}")
    for k, c in enumerate(CLASSES):
        sel = y == k
        nk = int(sel.sum())
        if nk == 0:
            print(f"  {c:<22}{0:>5}{'--':>18}{'--':>18}{'--':>9}")
            continue
        ka, kb = int(ca[sel].sum()), int(cb[sel].sum())
        sa, sb = ka / nk, kb / nk
        wa, wb_ = wilson(ka, nk), wilson(kb, nk)
        star = "  <<<" if c in ("Melanoma", "Basal_Cell_Carcinoma") else ""
        print(f"  {c:<22}{nk:>5}"
              f"{sa:8.4f} [{wa[0]:.3f},{wa[1]:.3f}]"
              f"{sb:8.4f} [{wb_[0]:.3f},{wb_[1]:.3f}]"
              f"{sb - sa:+9.4f}{star}")

    print(f"\nSAFETY-CRITICAL (Wilson 95% CI)")
    for c in ("Melanoma", "Basal_Cell_Carcinoma"):
        k = CLASSES.index(c)
        sel = y == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        ka, kb = int(ca[sel].sum()), int(cb[sel].sum())
        wa, wb_ = wilson(ka, nk), wilson(kb, nk)
        print(f"  {c} sensitivity  n={nk}")
        print(f"    A {ka}/{nk} = {ka / nk:.4f}   95% CI [{wa[0]:.4f}, {wa[1]:.4f}]")
        print(f"    B {kb}/{nk} = {kb / nk:.4f}   95% CI [{wb_[0]:.4f}, {wb_[1]:.4f}]")
        m = mcnemar(ca[sel], cb[sel])
        print(f"    paired McNemar on this class: b={m['b_A_right_B_wrong']} "
              f"c={m['c_A_wrong_B_right']}  exact p={m['p_exact']:.4f}")

    m = mcnemar(ca, cb)
    print(f"\nMcNemar's test (paired, same images, A vs B)")
    print(f"  both correct                     {m['both_correct']}")
    print(f"  both wrong                       {m['both_wrong']}")
    print(f"  b = A correct, B wrong           {m['b_A_right_B_wrong']}   (hair removal broke)")
    print(f"  c = A wrong,   B correct         {m['c_A_wrong_B_right']}   (hair removal fixed)")
    print(f"  discordant pairs n = b+c         {m['n_discordant']}")
    print(f"  chi2 (Yates-corrected, 1 df)     {m['chi2_cc']:.4f}   p = {m['p_chi2_cc']:.4f}")
    print(f"  exact binomial b vs b+c          p = {m['p_exact']:.4f}")

    flips = int(np.sum(preda != predb))
    print(f"\nlabel flips A->B (any direction): {flips}/{n} ({100 * flips / n:.2f}%)")
    maxd = np.abs(pa - pb).max(1)
    print(f"max per-class prob shift: mean {maxd.mean():.5f}  median "
          f"{np.median(maxd):.5f}  p95 {np.percentile(maxd, 95):.5f}  max {maxd.max():.5f}")


def main() -> None:
    if not RAW.exists():
        sys.exit(f"missing {RAW}")
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    print(f"loaded {len(rows)} paired rows from {RAW}")

    y = np.array([CLASSES.index(r["true_class"]) for r in rows])
    pa = np.array([[float(r[f"a_{c}"]) for c in CLASSES] for r in rows])
    pb = np.array([[float(r[f"b_{c}"]) for c in CLASSES] for r in rows])
    fired = np.array([r["hair_applied"] == "1" for r in rows])

    # sanity: recorded labels must match the recomputed argmax
    assert all(CLASSES[int(pa[i].argmax())] == r["pred_a"] for i, r in enumerate(rows))
    assert all(CLASSES[int(pb[i].argmax())] == r["pred_b"] for i, r in enumerate(rows))

    # ---- 1. firing rate ----
    print(f"\n{'=' * 78}\n1. remove_hair FIRING RATE\n{'=' * 78}")
    lo, hi = wilson(int(fired.sum()), len(rows))
    print(f"  overall applied=True  {int(fired.sum())}/{len(rows)} = "
          f"{fired.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  overall skipped       {int((~fired).sum())}/{len(rows)} = {(~fired).mean():.4f}")
    print(f"\n  {'class':<22}{'n':>5}{'fired':>7}{'rate':>9}{'95% CI':>20}")
    for k, c in enumerate(CLASSES):
        sel = y == k
        nk, kf = int(sel.sum()), int(fired[sel].sum())
        l, h = wilson(kf, nk)
        print(f"  {c:<22}{nk:>5}{kf:>7}{kf / nk:9.4f}   [{l:.4f}, {h:.4f}]")

    print(f"\n  reason the step declined, by bucket:")
    buckets: dict[str, int] = {}
    for r in rows:
        d = r["hair_note"] if "hair_note" in r else r.get("hair_detail", "")
        if r["hair_applied"] == "1":
            key = "APPLIED (inpainted)"
        elif d.startswith("skipped"):
            key = "skipped: mask > 10% (guard tripped)"
        elif "no hair" in d:
            key = "no hair detected (mask empty)"
        else:
            key = d
        buckets[key] = buckets.get(key, 0) + 1
    for k2, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {k2:<42}{v:>5}  ({100 * v / len(rows):5.2f}%)")

    # guard-trip mask fractions, for tuning context
    fracs = []
    for r in rows:
        d = r["hair_note"] if "hair_note" in r else r.get("hair_detail", "")
        if d.startswith("skipped"):
            try:
                fracs.append(float(d.split("covers")[1].split("%")[0]) / 100)
            except (IndexError, ValueError):
                pass
    if fracs:
        f = np.array(fracs)
        print(f"\n  mask fraction when the >10% guard tripped (n={len(f)}): "
              f"median {f.mean():.3f}/{np.median(f):.3f} mean/med, "
              f"min {f.min():.3f}, max {f.max():.3f}")

    # ---- 2/3. metrics ----
    block("2. WHOLE SET -- A (baseline) vs B (hair removal attempted on all)", y, pa, pb)
    block("3. FIRED SUBSET ONLY -- images where remove_hair reported applied=True",
          y[fired], pa[fired], pb[fired])
    block("3b. SKIPPED SUBSET (control: B must be identical to A here)",
          y[~fired], pa[~fired], pb[~fired])


if __name__ == "__main__":
    main()
