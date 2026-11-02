"""Analysis of the paired artifact-removal A/B experiment.

Reads `.nb1out/artifact_ab_raw.csv` (written by `artifact_ab_experiment.py`) and
reports firing rate by class, whole-set / fired-subset / skipped-subset metrics,
McNemar's test, Wilson CIs, every label flip, and the mask-coverage diagnostics that
show what MAX_MASK_FRACTION is suppressing.

Pure stdlib + numpy (scipy is not installed in this venv), so every statistic is
computed from first principles. Statistics helpers are the same ones used by
`hair_ab_analyze.py`, kept here verbatim so the two reports are comparable.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
MALIGNANT = {"Basal_Cell_Carcinoma", "Melanoma"}
RAW = ROOT / ".nb1out" / "artifact_ab_raw.csv"
MAX_MASK_FRACTION = 0.10


# ---------- statistics, from scratch ----------

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_two_sided_exact(b: int, n: int) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pmf[b]
    return min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-12)))


def chi2_sf_1df(x: float) -> float:
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """b = A right, B wrong (the step broke it); c = A wrong, B right (the step fixed it)."""
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


def q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


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
    print(f"\naccuracy   A(baseline)  {acc_a:7.4f}  [{la:.4f}, {ha:.4f}]  {int(ca.sum())}/{n}")
    print(f"           B(artifact)  {acc_b:7.4f}  [{lb:.4f}, {hb:.4f}]  {int(cb.sum())}/{n}")
    print(f"           delta B-A    {acc_b - acc_a:+7.4f}  ({100 * (acc_b - acc_a):+.2f} pp)")

    ma, pera = macro_auc(pa, y)
    mb, perb = macro_auc(pb, y)
    print(f"\nmacro AUC (one-vs-rest)   A {ma:.4f}   B {mb:.4f}   delta {mb - ma:+.4f}")
    print(f"  {'class':<22}{'AUC A':>9}{'AUC B':>9}{'delta':>9}")
    for c in CLASSES:
        print(f"  {c:<22}{pera[c]:9.4f}{perb[c]:9.4f}{perb[c] - pera[c]:+9.4f}")

    print("\nper-class sensitivity / recall")
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
        star = "  <<<" if c in MALIGNANT else ""
        print(f"  {c:<22}{nk:>5}"
              f"{sa:8.4f} [{wa[0]:.3f},{wa[1]:.3f}]"
              f"{sb:8.4f} [{wb_[0]:.3f},{wb_[1]:.3f}]"
              f"{sb - sa:+9.4f}{star}")

    print("\nSAFETY-CRITICAL (Wilson 95% CI)")
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
    print("\nMcNemar's test (paired, same images, A vs B)")
    print(f"  both correct                     {m['both_correct']}")
    print(f"  both wrong                       {m['both_wrong']}")
    print(f"  b = A correct, B wrong           {m['b_A_right_B_wrong']}   (artifact removal broke)")
    print(f"  c = A wrong,   B correct         {m['c_A_wrong_B_right']}   (artifact removal fixed)")
    print(f"  discordant pairs n = b+c         {m['n_discordant']}")
    print(f"  chi2 (Yates-corrected, 1 df)     {m['chi2_cc']:.4f}   p = {m['p_chi2_cc']:.4f}")
    print(f"  exact binomial b vs b+c          p = {m['p_exact']:.4f}")

    flips = int(np.sum(preda != predb))
    print(f"\nlabel flips A->B (any direction): {flips}/{n} ({100 * flips / n:.2f}%)")
    maxd = np.abs(pa - pb).max(1)
    print(f"max per-class prob shift: mean {maxd.mean():.5f}  median "
          f"{np.median(maxd):.5f}  p95 {q(maxd, 95):.5f}  max {maxd.max():.5f}")


def main() -> None:
    if not RAW.exists():
        sys.exit(f"missing {RAW}")
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    print(f"loaded {len(rows)} paired rows from {RAW}")

    y = np.array([CLASSES.index(r["true_class"]) for r in rows])
    pa = np.array([[float(r[f"a_{c}"]) for c in CLASSES] for r in rows])
    pb = np.array([[float(r[f"b_{c}"]) for c in CLASSES] for r in rows])
    fired = np.array([r["artifact_applied"] == "1" for r in rows])
    fc = np.array([float(r["frac_combined"]) for r in rows])
    fi = np.array([float(r["frac_ink"]) for r in rows])
    fg = np.array([float(r["frac_glare"]) for r in rows])
    fr = np.array([float(r["frac_ruler"]) for r in rows])

    assert all(CLASSES[int(pa[i].argmax())] == r["pred_a"] for i, r in enumerate(rows))
    assert all(CLASSES[int(pb[i].argmax())] == r["pred_b"] for i, r in enumerate(rows))

    # ---- 1. firing rate ----
    print(f"\n{'=' * 78}\n1. remove_artifacts FIRING RATE\n{'=' * 78}")
    lo, hi = wilson(int(fired.sum()), len(rows))
    print(f"  overall applied=True  {int(fired.sum())}/{len(rows)} = "
          f"{fired.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  overall skipped       {int((~fired).sum())}/{len(rows)} = {(~fired).mean():.4f}")
    print(f"\n  {'class':<22}{'n':>5}{'fired':>7}{'rate':>9}{'95% CI':>22}")
    for k, c in enumerate(CLASSES):
        sel = y == k
        nk, kf = int(sel.sum()), int(fired[sel].sum())
        l, h = wilson(kf, nk)
        print(f"  {c:<22}{nk:>5}{kf:>7}{kf / nk:9.4f}   [{l:.4f}, {h:.4f}]")

    print("\n  reason the step declined, by bucket:")
    buckets: dict[str, int] = {}
    for r in rows:
        d = r["artifact_note"]
        if r["artifact_applied"] == "1":
            key = "APPLIED (inpainted)"
        elif d.startswith("skipped"):
            key = f"skipped: mask > {MAX_MASK_FRACTION:.0%} (guard tripped)"
        elif "no artifacts" in d:
            key = "no artifacts detected (mask empty)"
        else:
            key = d
        buckets[key] = buckets.get(key, 0) + 1
    for k2, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {k2:<42}{v:>5}  ({100 * v / len(rows):5.2f}%)")

    # ---- 2. metrics ----
    block("2. WHOLE SET -- A (baseline) vs B (artifact removal attempted on all)", y, pa, pb)
    block("3. FIRED SUBSET ONLY -- images where remove_artifacts reported applied=True",
          y[fired], pa[fired], pb[fired])
    block("4. SKIPPED SUBSET (control: B must be identical to A here)",
          y[~fired], pa[~fired], pb[~fired])

    # ---- 5. control check ----
    print(f"\n{'=' * 78}\n5. CONTROL CHECK -- determinism on non-fired images\n{'=' * 78}")
    if int((~fired).sum()):
        d = np.abs(pa[~fired] - pb[~fired])
        print(f"  n skipped = {int((~fired).sum())}")
        print(f"  max absolute probability difference over all classes: {d.max():.8f}")
        print(f"  images with ANY nonzero difference: {int((d.max(1) > 0).sum())}")
        print(f"  label flips among skipped images: "
              f"{int((pa[~fired].argmax(1) != pb[~fired].argmax(1)).sum())}")
        print("  -> 0.00000000 means the scoring path is deterministic and every flip "
              "observed above is a genuine inpainting effect.")

    # ---- 6. every flip, listed ----
    print(f"\n{'=' * 78}\n6. EVERY LABEL FLIP, LISTED\n{'=' * 78}")
    flip_idx = np.where(pa.argmax(1) != pb.argmax(1))[0]
    if len(flip_idx) == 0:
        print("  none -- artifact removal changed no prediction on any of the 565 images.")
    for i in flip_idx:
        r = rows[i]
        la_, lb_ = CLASSES[int(pa[i].argmax())], CLASSES[int(pb[i].argmax())]
        img_id = Path(r["path"]).stem
        truth = r["true_class"]
        ok_a, ok_b = la_ == truth, lb_ == truth
        verdict = ("BROKEN (was right, now wrong)" if ok_a and not ok_b
                   else "FIXED (was wrong, now right)" if ok_b and not ok_a
                   else "wrong both ways")
        worst = ""
        if truth in MALIGNANT and la_ == truth and lb_ not in MALIGNANT:
            worst = "   *** WORST CASE: true malignant, confident->benign ***"
        elif truth in MALIGNANT and ok_a and not ok_b:
            worst = "   ** malignant misclassified **"
        print(f"\n  {img_id}  true={truth}")
        print(f"    A: {la_:<22} p={pa[i].max():.4f}      (p_true={pa[i][CLASSES.index(truth)]:.4f})")
        print(f"    B: {lb_:<22} p={pb[i].max():.4f}      (p_true={pb[i][CLASSES.index(truth)]:.4f})")
        print(f"    mask: combined {float(r['frac_combined']):.4%} "
              f"(ink {float(r['frac_ink']):.4%}, glare {float(r['frac_glare']):.4%}, "
              f"ruler {float(r['frac_ruler']):.4%})")
        print(f"    note: {r['artifact_note']}")
        print(f"    -> {verdict}{worst}")

    # ---- 7. mask diagnostics ----
    print(f"\n{'=' * 78}\n7. MASK COVERAGE DIAGNOSTICS\n{'=' * 78}")

    def dist(name: str, a: np.ndarray) -> None:
        if not len(a):
            print(f"  {name:<34} n=0")
            return
        print(f"  {name:<34} n={len(a):>4}  min {a.min():7.4%}  p25 {q(a, 25):7.4%}  "
              f"median {q(a, 50):7.4%}  p75 {q(a, 75):7.4%}  max {a.max():7.4%}")

    print("\n  combined mask fraction of frame:")
    dist("ALL images", fc)
    dist("FIRED (inpainted)", fc[fired])
    dist("SKIPPED (any reason)", fc[~fired])
    guard = np.array([r["artifact_note"].startswith("skipped") for r in rows])
    empty = np.array([float(r["frac_combined"]) == 0.0 for r in rows])
    dist(f"SKIPPED by >{MAX_MASK_FRACTION:.0%} guard", fc[guard])
    dist("SKIPPED mask empty", fc[empty])

    print("\n  per-detector contribution (fraction of frame), ALL images:")
    for nm, a in (("ink", fi), ("glare", fg), ("ruler", fr)):
        dist(nm, a)
    print("\n  per-detector, FIRED images only:")
    for nm, a in (("ink", fi), ("glare", fg), ("ruler", fr)):
        dist(nm, a[fired])
    print("\n  per-detector, guard-tripped images only:")
    for nm, a in (("ink", fi), ("glare", fg), ("ruler", fr)):
        dist(nm, a[guard])

    print("\n  detector that dominates each image's mask (argmax of the three):")
    stack = np.stack([fi, fg, fr], 1)
    names = ["ink", "glare", "ruler"]
    for sub, lbl in ((np.ones(len(rows), bool), "all"), (fired, "fired"), (guard, "guard-tripped")):
        if not sub.sum():
            continue
        dom = stack[sub].argmax(1)
        any_mask = stack[sub].sum(1) > 0
        counts = {names[j]: int(((dom == j) & any_mask).sum()) for j in range(3)}
        counts["no mask at all"] = int((~any_mask).sum())
        print(f"    {lbl:<16}" + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\n  what the >{MAX_MASK_FRACTION:.0%} guard is suppressing:")
    print(f"    images the guard stopped from being inpainted: {int(guard.sum())} "
          f"({100 * guard.mean():.2f}% of {len(rows)})")
    if guard.sum():
        g = fc[guard]
        print(f"    their mask coverage: median {q(g, 50):.2%}, mean {g.mean():.2%}, max {g.max():.2%}")
        for thr in (0.25, 0.50, 0.75):
            print(f"      >= {thr:.0%} of the frame: {int((g >= thr).sum())} images")
        print(f"    without the guard, {int(guard.sum()) + int(fired.sum())} of {len(rows)} "
              f"images ({100 * (guard.sum() + fired.sum()) / len(rows):.1f}%) would be inpainted.")
        print("\n    guard trips by class:")
        for k, c in enumerate(CLASSES):
            sel = y == k
            print(f"      {c:<22}{int(guard[sel].sum()):>4}/{int(sel.sum()):<4} "
                  f"= {guard[sel].mean():.2%}   median mask "
                  f"{q(fc[sel & guard], 50) if (sel & guard).sum() else float('nan'):.2%}")

    print("\n  firing rate vs mask size, for context on how close the fired images "
          "sit to the cutoff:")
    if fired.sum():
        f_ = fc[fired]
        print(f"    fired-image masks as a share of the {MAX_MASK_FRACTION:.0%} cutoff: "
              f"median {q(f_, 50) / MAX_MASK_FRACTION:.2f}x, p90 {q(f_, 90) / MAX_MASK_FRACTION:.2f}x")
        print(f"    fired images with mask < 1% of frame: {int((f_ < 0.01).sum())}/{int(fired.sum())}")


if __name__ == "__main__":
    main()
