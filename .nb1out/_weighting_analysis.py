"""Ensemble weighting analysis: should ECS keep equal weights, judged on melanoma sensitivity?

Loads the cached per-backbone logits (same conventions as .nb1out/_agree.py and
_pairedC.py), reconstructs the 7 equal-weight configs, and runs:
  1. Wilson 95% CIs on melanoma sensitivity
  2. McNemar (exact binomial) ECS vs ES, melanoma-only and all-565
  3. Weight grid search (test-set fitted -> optimistically biased)
  4. Robustness on the contaminated 727
  5. Validation-set replication

No scipy in the venv, so the exact binomial and Wilson interval are implemented here.
"""
import csv
import itertools
import math
import sys
from pathlib import PurePosixPath as P

import numpy as np

D = r"F:\Projects\SkinSure\.nb1out"
MAN = r"C:\Users\Cipher\Downloads\test_clean_manifest.csv"
BACKBONES = ["efficientnet_b1", "convnext_tiny", "swin_tiny"]
CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
MEL = 3
BCC = 0
K = 5
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def exact_binom_two_sided(b, c):
    """Exact two-sided binomial test on the discordant pairs, p=0.5.

    With p=0.5 the distribution is symmetric, so the two-sided p-value is
    simply twice the smaller tail, capped at 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(binom_pmf(i, n, 0.5) for i in range(k + 1))
    return min(1.0, 2 * tail)


def mcnemar_chi2(b, c, correction=True):
    if b + c == 0:
        return 0.0
    num = abs(b - c) - (1 if correction else 0)
    num = max(num, 0) if correction else abs(b - c)
    return num ** 2 / (b + c)


def chi2_sf_1df(x):
    """Survival function of chi-square with 1 df = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def auc_binary(score, pos):
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
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


def macro_auc(p, y):
    return float(np.mean([auc_binary(p[:, k], y == k) for k in range(K)]))


def mcc(p, y):
    pred = p.argmax(1)
    C = np.zeros((K, K), dtype=float)
    np.add.at(C, (y, pred), 1)
    t, q = C.sum(1), C.sum(0)
    c, s = np.trace(C), C.sum()
    num = c * s - t @ q
    den = math.sqrt(s * s - q @ q) * math.sqrt(s * s - t @ t)
    return float(num / den) if den else 0.0


def balanced_acc(p, y):
    pred = p.argmax(1)
    return float(np.mean([(pred[y == k] == k).mean() for k in range(K)]))


# ---------------------------------------------------------------- data load
def load(split, manifest=None):
    Z = {k: np.load(f"{D}\\cache_{k}_{split}.npz", allow_pickle=True) for k in BACKBONES}
    paths = Z[BACKBONES[0]]["paths"]
    labels = Z[BACKBONES[0]]["labels"]
    # the three caches must be row-aligned on the same paths
    for k in BACKBONES[1:]:
        assert np.array_equal(Z[k]["paths"], paths), f"path mismatch {k}"
        assert np.array_equal(Z[k]["labels"], labels), f"label mismatch {k}"
        assert list(Z[k]["classes"]) == CLASSES, f"class order {k}: {list(Z[k]['classes'])}"
    if manifest:
        keep = {f"{P(r['path']).parent.name}/{P(r['path']).name}"
                for r in csv.DictReader(open(manifest, encoding="utf-8-sig"))}
        mask = np.array([f"{P(str(p)).parent.name}/{P(str(p)).name}" in keep for p in paths])
    else:
        mask = np.ones(len(paths), dtype=bool)
    y = labels[mask]
    probs = {k: Z[k]["probs"][mask] for k in BACKBONES}
    return y, probs, paths[mask]


CONFIGS = {
    "E": ("efficientnet_b1",),
    "C": ("convnext_tiny",),
    "S": ("swin_tiny",),
    "EC": ("efficientnet_b1", "convnext_tiny"),
    "ES": ("efficientnet_b1", "swin_tiny"),
    "CS": ("convnext_tiny", "swin_tiny"),
    "ECS": ("efficientnet_b1", "convnext_tiny", "swin_tiny"),
}
LABELS = {"E": "EfficientNet-B1", "C": "ConvNeXt-Tiny", "S": "Swin-Tiny",
          "EC": "EfficientNet + ConvNeXt", "ES": "EfficientNet + Swin",
          "CS": "ConvNeXt + Swin", "ECS": "EfficientNet + ConvNeXt + Swin"}


def build(probs, members):
    return sum(probs[m] for m in members) / len(members)


# =========================================================== 0. SANITY CHECK
y, PR, paths = load("test", MAN)
n = len(y)
say(f"# Ensemble weighting analysis\n")
say(f"Clean test set: n={n} (expected 565 of 727)")
say(f"Class counts: " + ", ".join(f"{c}={int((y==i).sum())}" for i, c in enumerate(CLASSES)))

ens = {name: build(PR, mem) for name, mem in CONFIGS.items()}

ref = {}
for r in csv.DictReader(open(f"{D}\\ablation_table.csv", encoding="utf-8-sig")):
    ref[r["config"]] = r

say("\n## 0. Reproduction check against ablation_table.csv\n")
say("| config | acc (mine) | acc (NB1) | delta | macroAUC (mine) | macroAUC (NB1) | delta | MCC (mine) | MCC (NB1) | delta |")
say("|---|---|---|---|---|---|---|---|---|---|")
sanity_rows = []
max_delta = 0.0
for name in CONFIGS:
    p = ens[name]
    a, au, m = (p.argmax(1) == y).mean(), macro_auc(p, y), mcc(p, y)
    ra, rau, rm = float(ref[name]["accuracy"]), float(ref[name]["macro_auc"]), float(ref[name]["mcc"])
    max_delta = max(max_delta, abs(a - ra), abs(au - rau), abs(m - rm))
    say(f"| {name} | {a:.6f} | {ra:.6f} | {a-ra:+.2e} | {au:.6f} | {rau:.6f} | {au-rau:+.2e} | {m:.6f} | {rm:.6f} | {m-rm:+.2e} |")
    sanity_rows.append(dict(config=name, acc_mine=a, acc_nb1=ra, auc_mine=au, auc_nb1=rau,
                            mcc_mine=m, mcc_nb1=rm))
say(f"\nMax absolute deviation across all 21 cells: **{max_delta:.2e}**. "
    + ("Reconstruction matches NB1 exactly; everything downstream is trustworthy."
       if max_delta < 1e-9 else "MISMATCH -- investigate before trusting downstream results."))

with open(f"{D}\\weighting_sanity_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(sanity_rows[0]))
    w.writeheader()
    w.writerows(sanity_rows)

# ====================================================== 1. WILSON CIs
say("\n## 1. Wilson 95% CIs on melanoma sensitivity (clean 565)\n")
mel_mask = y == MEL
n_mel = int(mel_mask.sum())
say(f"Melanoma cases: n={n_mel}\n")
say("| config | TP | FN | sensitivity | Wilson 95% CI | CI width |")
say("|---|---|---|---|---|---|")
wilson_rows = []
sens = {}
for name in CONFIGS:
    pred = ens[name].argmax(1)
    tp = int((pred[mel_mask] == MEL).sum())
    p, lo, hi = wilson(tp, n_mel)
    sens[name] = (tp, p)
    say(f"| {name} | {tp} | {n_mel-tp} | {p*100:.2f}% | [{lo*100:.2f}%, {hi*100:.2f}%] | {(hi-lo)*100:.2f} pp |")
    wilson_rows.append(dict(config=name, label=LABELS[name], n_melanoma=n_mel, tp=tp,
                            fn=n_mel - tp, sensitivity=p, wilson_lo=lo, wilson_hi=hi))

# also BCC
say(f"\nSecondary target -- BCC sensitivity (n={int((y==BCC).sum())}):\n")
say("| config | TP | FN | sensitivity | Wilson 95% CI |")
say("|---|---|---|---|---|")
bcc_mask = y == BCC
n_bcc = int(bcc_mask.sum())
for name in CONFIGS:
    pred = ens[name].argmax(1)
    tp = int((pred[bcc_mask] == BCC).sum())
    p, lo, hi = wilson(tp, n_bcc)
    say(f"| {name} | {tp} | {n_bcc-tp} | {p*100:.2f}% | [{lo*100:.2f}%, {hi*100:.2f}%] |")
    for r in wilson_rows:
        if r["config"] == name:
            r.update(bcc_tp=tp, bcc_n=n_bcc, bcc_sens=p, bcc_lo=lo, bcc_hi=hi)

with open(f"{D}\\weighting_wilson_ci.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(wilson_rows[0]))
    w.writeheader()
    w.writerows(wilson_rows)

lo_ecs = wilson(sens["ECS"][0], n_mel)[1]
hi_ecs = wilson(sens["ECS"][0], n_mel)[2]
lo_es = wilson(sens["ES"][0], n_mel)[1]
hi_es = wilson(sens["ES"][0], n_mel)[2]
say(f"\n**Overlap:** ECS CI [{lo_ecs*100:.2f}%, {hi_ecs*100:.2f}%] and ES CI "
    f"[{lo_es*100:.2f}%, {hi_es*100:.2f}%] overlap over "
    f"[{max(lo_ecs,lo_es)*100:.2f}%, {min(hi_ecs,hi_es)*100:.2f}%] -- "
    f"a span of {(min(hi_ecs,hi_es)-max(lo_ecs,lo_es))*100:.2f} pp.")
say("\nIn fact every one of the seven intervals overlaps every other: the lowest "
    f"upper bound is {min(wilson(sens[c][0], n_mel)[2] for c in CONFIGS)*100:.2f}% and the highest "
    f"lower bound is {max(wilson(sens[c][0], n_mel)[1] for c in CONFIGS)*100:.2f}%, "
    "so no pair of configurations is separated on melanoma sensitivity.")

# ====================================================== 2. McNEMAR
say("\n## 2. McNemar's test, ECS vs ES\n")


def mcnemar(a_name, b_name, sel, label, store):
    pa = ens[a_name].argmax(1)[sel]
    pb = ens[b_name].argmax(1)[sel]
    yy = y[sel]
    ca, cb = pa == yy, pb == yy
    b = int((ca & ~cb).sum())   # a right, b wrong
    c = int((~ca & cb).sum())   # a wrong, b right
    both = int((ca & cb).sum())
    neither = int((~ca & ~cb).sum())
    chi_cc = mcnemar_chi2(b, c, True)
    chi_nc = mcnemar_chi2(b, c, False)
    p_exact = exact_binom_two_sided(b, c)
    say(f"### {label}\n")
    say(f"n = {int(sel.sum())} paired images")
    say(f"```")
    say(f"                       {b_name} correct   {b_name} wrong")
    say(f"  {a_name} correct        {both:>5}            {b:>5}   <- b")
    say(f"  {a_name} wrong          {c:>5}            {neither:>5}")
    say(f"                            ^ c")
    say(f"```")
    say(f"- Discordant pairs: b={b} ({a_name} right, {b_name} wrong), c={c} ({a_name} wrong, {b_name} right), b+c={b+c}")
    say(f"- McNemar chi2 (continuity-corrected) = {chi_cc:.4f}, 1 df, asymptotic p = {chi2_sf_1df(chi_cc):.4f}")
    say(f"- McNemar chi2 (uncorrected) = {chi_nc:.4f}, asymptotic p = {chi2_sf_1df(chi_nc):.4f}")
    say(f"- **Exact two-sided binomial p = {p_exact:.4f}**"
        + ("  (b+c small, so the exact test is the one to quote)" if b + c < 25 else ""))
    store.append(dict(comparison=f"{a_name}_vs_{b_name}", subset=label, n=int(sel.sum()),
                      b=b, c=c, both_correct=both, neither_correct=neither,
                      chi2_cc=chi_cc, p_chi2_cc=chi2_sf_1df(chi_cc),
                      chi2_uncorrected=chi_nc, p_chi2_uncorrected=chi2_sf_1df(chi_nc),
                      p_exact_binomial=p_exact))
    return b, c, p_exact


mc_rows = []
say("Convention: b = ECS correct & ES wrong; c = ECS wrong & ES correct.\n")
mcnemar("ECS", "ES", mel_mask, "Melanoma cases only", mc_rows)
say("")
mcnemar("ECS", "ES", np.ones(n, dtype=bool), "All 565 images (overall accuracy)", mc_rows)

# extra context: ECS vs the best-sensitivity alternatives
say("\n### Supporting comparisons (same protocol)\n")
for a, b_ in [("ECS", "CS"), ("ECS", "C"), ("ES", "CS")]:
    mcnemar(a, b_, mel_mask, f"Melanoma only, {a} vs {b_}", mc_rows)
    say("")

with open(f"{D}\\weighting_mcnemar.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(mc_rows[0]))
    w.writeheader()
    w.writerows(mc_rows)

# ====================================================== 3. WEIGHT GRID
say("\n## 3. Weight grid search (FITTED ON THE TEST SET -- optimistically biased)\n")
say("> **Read this before using any number in this section.** The grid below is scored on "
    "the same 565 images it selects on. Whatever it picks is the maximum of ~230 correlated "
    "estimates, so its apparent score is biased upward by the selection itself. It answers "
    "*is there headroom?*, not *what weights should we ship?*. A weight vector taken from here "
    "and deployed would be expected to regress toward the equal-weight result on new data.\n")

step = 0.1
grid = [round(i * step, 1) for i in range(11)]
E, C_, S = PR["efficientnet_b1"], PR["convnext_tiny"], PR["swin_tiny"]
rows = []
seen = set()
for we, wc, ws in itertools.product(grid, repeat=3):
    t = we + wc + ws
    if t == 0:
        continue
    key = (round(we / t, 6), round(wc / t, 6), round(ws / t, 6))
    if key in seen:
        continue
    seen.add(key)
    a, b_, c_ = key
    p = a * E + b_ * C_ + c_ * S
    pred = p.argmax(1)
    tp = int((pred[mel_mask] == MEL).sum())
    rows.append(dict(w_eff=a, w_conv=b_, w_swin=c_,
                     mel_tp=tp, mel_sens=tp / n_mel,
                     bcc_sens=float((pred[bcc_mask] == BCC).mean()),
                     accuracy=float((pred == y).mean()),
                     balanced_acc=balanced_acc(p, y),
                     macro_auc=macro_auc(p, y), mcc=mcc(p, y)))
say(f"Grid: each weight in 0.0..1.0 step 0.1, normalised to sum 1, all-zero skipped, "
    f"duplicates after normalisation removed -> **{len(rows)} distinct weight vectors**.\n")

with open(f"{D}\\weighting_grid_search.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: -r["mel_sens"]))


def eq(r):
    return abs(r["w_eff"] - 1 / 3) < 1e-6 and abs(r["w_conv"] - 1 / 3) < 1e-6


eqrow = next(r for r in rows if eq(r))


def top(metric, k=10):
    srt = sorted(rows, key=lambda r: (-r[metric], -r["macro_auc"]))
    say(f"\n### (a-c) Top {k} by {metric}\n")
    say("| rank | w_E | w_C | w_S | mel sens | mel TP/124 | BCC sens | accuracy | macro AUC | MCC |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(srt[:k], 1):
        say(f"| {i} | {r['w_eff']:.3f} | {r['w_conv']:.3f} | {r['w_swin']:.3f} | "
            f"{r['mel_sens']*100:.2f}% | {r['mel_tp']}/{n_mel} | {r['bcc_sens']*100:.2f}% | "
            f"{r['accuracy']*100:.2f}% | {r['macro_auc']:.5f} | {r['mcc']:.4f} |")
    best = srt[0]
    say(f"\nBest {metric} = {best[metric]:.6f} at weights "
        f"(E={best['w_eff']:.3f}, C={best['w_conv']:.3f}, S={best['w_swin']:.3f}); "
        f"equal weights give {eqrow[metric]:.6f} "
        f"(gap {best[metric]-eqrow[metric]:+.6f}).")
    return srt


say(f"\nEqual-weight ECS reference row: mel sens {eqrow['mel_sens']*100:.2f}% "
    f"({eqrow['mel_tp']}/{n_mel}), acc {eqrow['accuracy']*100:.2f}%, "
    f"macro AUC {eqrow['macro_auc']:.5f}, MCC {eqrow['mcc']:.4f}.")

srt_sens = top("mel_sens")
srt_auc = top("macro_auc")
srt_acc = top("accuracy")

# how many weightings reach each melanoma TP level
say("\n### Distribution of melanoma sensitivity across the grid\n")
say("| melanoma TP | sensitivity | # weight vectors | share of grid |")
say("|---|---|---|---|")
import collections
cnt = collections.Counter(r["mel_tp"] for r in rows)
for tp in sorted(cnt, reverse=True):
    say(f"| {tp}/{n_mel} | {tp/n_mel*100:.2f}% | {cnt[tp]} | {cnt[tp]/len(rows)*100:.1f}% |")

# does any 3-model weighting beat the best 2-model drop?
_three = [r for r in rows if min(r["w_eff"], r["w_conv"], r["w_swin"]) > 0]
best_sens_3 = max(_three, key=lambda r: (r["mel_sens"], r["macro_auc"]))
say(f"\n### Does a genuinely 3-backbone weighting (all weights > 0) beat dropping one?\n")
say(f"- Best all-three-nonzero melanoma sensitivity: {best_sens_3['mel_sens']*100:.2f}% "
    f"({best_sens_3['mel_tp']}/{n_mel}) at (E={best_sens_3['w_eff']:.3f}, "
    f"C={best_sens_3['w_conv']:.3f}, S={best_sens_3['w_swin']:.3f}), "
    f"macro AUC {best_sens_3['macro_auc']:.5f}, acc {best_sens_3['accuracy']*100:.2f}%.")
es_row = next(r for r in rows if abs(r["w_conv"]) < 1e-9 and abs(r["w_eff"] - 0.5) < 1e-6)
say(f"- ES (the drop-one champion on sensitivity): {es_row['mel_sens']*100:.2f}% "
    f"({es_row['mel_tp']}/{n_mel}), macro AUC {es_row['macro_auc']:.5f}, "
    f"acc {es_row['accuracy']*100:.2f}%.")

# ====================================================== 4. CONTAMINATED 727
say("\n## 4. Robustness -- contaminated 727 basis (all cached rows, no manifest filter)\n")
y2, PR2, _ = load("test", None)
ens2 = {k: build(PR2, m) for k, m in CONFIGS.items()}
mel2 = y2 == MEL
n_mel2 = int(mel2.sum())
say(f"n={len(y2)}, melanoma n={n_mel2}\n")
say("| config | mel TP | mel sensitivity | Wilson 95% CI | accuracy |")
say("|---|---|---|---|---|")
contam_rows = []
for name in CONFIGS:
    pred = ens2[name].argmax(1)
    tp = int((pred[mel2] == MEL).sum())
    p, lo, hi = wilson(tp, n_mel2)
    acc = float((pred == y2).mean())
    say(f"| {name} | {tp}/{n_mel2} | {p*100:.2f}% | [{lo*100:.2f}%, {hi*100:.2f}%] | {acc*100:.2f}% |")
    contam_rows.append(dict(basis="contaminated_727", config=name, n_mel=n_mel2, mel_tp=tp,
                            mel_sens=p, lo=lo, hi=hi, accuracy=acc))

pa = ens2["ECS"].argmax(1)[mel2] == MEL
pb = ens2["ES"].argmax(1)[mel2] == MEL
b2, c2 = int((pa & ~pb).sum()), int((~pa & pb).sum())
say(f"\nMcNemar ECS vs ES on melanoma, contaminated basis: b={b2}, c={c2}, "
    f"exact two-sided p = {exact_binom_two_sided(b2, c2):.4f}")
contam_rows.append(dict(basis="contaminated_727", config="MCNEMAR_ECS_vs_ES_melanoma",
                        n_mel=n_mel2, mel_tp=b2, mel_sens=c2, lo=exact_binom_two_sided(b2, c2),
                        hi="", accuracy=""))

# ====================================================== 5. VALIDATION SET
say("\n## 5. Replication on the validation split\n")
yv, PRv, _ = load("val", None)
ensv = {k: build(PRv, m) for k, m in CONFIGS.items()}
melv = yv == MEL
n_melv = int(melv.sum())
say(f"`cache_*_val.npz` exists: n={len(yv)}, melanoma n={n_melv}. "
    "No clean manifest covers val, so this is the full (unfiltered) validation split -- "
    "the analogue of the contaminated basis, not the clean 565.\n")
say("| config | mel TP | mel sensitivity | Wilson 95% CI | accuracy | macro AUC |")
say("|---|---|---|---|---|---|")
val_rows = []
for name in CONFIGS:
    p_ = ensv[name]
    pred = p_.argmax(1)
    tp = int((pred[melv] == MEL).sum())
    pr, lo, hi = wilson(tp, n_melv)
    say(f"| {name} | {tp}/{n_melv} | {pr*100:.2f}% | [{lo*100:.2f}%, {hi*100:.2f}%] | "
        f"{(pred==yv).mean()*100:.2f}% | {macro_auc(p_, yv):.5f} |")
    val_rows.append(dict(basis="val_full", config=name, n_mel=n_melv, mel_tp=tp, mel_sens=pr,
                         lo=lo, hi=hi, accuracy=float((pred == yv).mean()),
                         macro_auc=macro_auc(p_, yv)))

pav = ensv["ECS"].argmax(1)[melv] == MEL
pbv = ensv["ES"].argmax(1)[melv] == MEL
bv, cv = int((pav & ~pbv).sum()), int((~pav & pbv).sum())
pv = exact_binom_two_sided(bv, cv)
say(f"\nMcNemar ECS vs ES on validation melanoma: b={bv} (ECS right, ES wrong), "
    f"c={cv} (ECS wrong, ES right), exact two-sided p = {pv:.4f}")
say(f"\nValidation on all {len(yv)} images:")
pav2 = ensv["ECS"].argmax(1) == yv
pbv2 = ensv["ES"].argmax(1) == yv
bv2, cv2 = int((pav2 & ~pbv2).sum()), int((~pav2 & pbv2).sum())
say(f"  b={bv2}, c={cv2}, exact two-sided p = {exact_binom_two_sided(bv2, cv2):.4f}")

with open(f"{D}\\weighting_robustness.csv", "w", newline="", encoding="utf-8") as f:
    fn = ["basis", "config", "n_mel", "mel_tp", "mel_sens", "lo", "hi", "accuracy", "macro_auc"]
    w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
    w.writeheader()
    w.writerows(contam_rows + val_rows)

# direction-of-effect summary across the three bases
say("\n### Direction of the ES-over-ECS melanoma advantage across all three bases\n")
say("| basis | n melanoma | ECS sens | ES sens | ES - ECS | McNemar b / c | exact p |")
say("|---|---|---|---|---|---|---|")
for lbl, yy, ee, mm in [("clean test 565", y, ens, mel_mask),
                        ("contaminated test 727", y2, ens2, mel2),
                        ("validation 748", yv, ensv, melv)]:
    nm = int(mm.sum())
    a_ok = ee["ECS"].argmax(1)[mm] == MEL
    b_ok = ee["ES"].argmax(1)[mm] == MEL
    bb, cc = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    sa, sb = a_ok.mean(), b_ok.mean()
    say(f"| {lbl} | {nm} | {sa*100:.2f}% | {sb*100:.2f}% | {(sb-sa)*100:+.2f} pp | "
        f"{bb} / {cc} | {exact_binom_two_sided(bb, cc):.4f} |")

with open(__file__ + ".out.md",
          "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[body written]")
