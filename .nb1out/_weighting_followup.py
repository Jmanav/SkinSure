"""Follow-ups: pooled test+val McNemar, which melanomas are missed, and the
sensitivity/accuracy trade frontier."""
import csv
import math
from pathlib import PurePosixPath as P

import numpy as np

D = r"F:\Projects\SkinSure\.nb1out"
MAN = r"C:\Users\Cipher\Downloads\test_clean_manifest.csv"
B = ["efficientnet_b1", "convnext_tiny", "swin_tiny"]
CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
MEL = 3
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def binom_pmf(k, n):
    return math.comb(n, k) * 0.5 ** n


def exact2(b, c):
    nn = b + c
    if nn == 0:
        return 1.0
    return min(1.0, 2 * sum(binom_pmf(i, nn) for i in range(min(b, c) + 1)))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(split, man=None):
    Z = {k: np.load(f"{D}\\cache_{k}_{split}.npz", allow_pickle=True) for k in B}
    paths, labels = Z[B[0]]["paths"], Z[B[0]]["labels"]
    if man:
        keep = {f"{P(r['path']).parent.name}/{P(r['path']).name}"
                for r in csv.DictReader(open(man, encoding="utf-8-sig"))}
        m = np.array([f"{P(str(p)).parent.name}/{P(str(p)).name}" in keep for p in paths])
    else:
        m = np.ones(len(paths), bool)
    return labels[m], {k: Z[k]["probs"][m] for k in B}, paths[m]


y, PR, pt = load("test", MAN)
yv, PRv, ptv = load("val", None)


def mk(pr, mem):
    return sum(pr[m] for m in mem) / len(mem)


ECS = mk(PR, B)
ES = mk(PR, ["efficientnet_b1", "swin_tiny"])
ECSv = mk(PRv, B)
ESv = mk(PRv, ["efficientnet_b1", "swin_tiny"])

say("## 6. Pooled evidence and the identity of the misses\n")

# ---- pooled test(clean 565) + val, melanoma only. Independent splits, so the
# discordant counts add.
m1, m2 = y == MEL, yv == MEL
a1, b1 = ECS.argmax(1)[m1] == MEL, ES.argmax(1)[m1] == MEL
a2, b2 = ECSv.argmax(1)[m2] == MEL, ESv.argmax(1)[m2] == MEL
B_ = int((a1 & ~b1).sum()) + int((a2 & ~b2).sum())
C_ = int((~a1 & b1).sum()) + int((~a2 & b2).sum())
say(f"### Pooled melanoma cases, clean test 565 + full validation 748\n")
say(f"These are disjoint image sets, so the discordant pairs pool additively.")
say(f"- n melanoma pooled = {int(m1.sum())} + {int(m2.sum())} = {int(m1.sum())+int(m2.sum())}")
say(f"- b (ECS right, ES wrong) = {B_};  c (ECS wrong, ES right) = {C_};  b+c = {B_+C_}")
say(f"- **Exact two-sided binomial p = {exact2(B_, C_):.4f}**  (one-sided favouring ES: "
    f"{sum(binom_pmf(i, B_+C_) for i in range(B_+1)):.4f})")
sa = (int(a1.sum()) + int(a2.sum())) / (int(m1.sum()) + int(m2.sum()))
sb = (int(b1.sum()) + int(b2.sum())) / (int(m1.sum()) + int(m2.sum()))
say(f"- Pooled ECS sensitivity {sa*100:.2f}% ({int(a1.sum())+int(a2.sum())}/"
    f"{int(m1.sum())+int(m2.sum())}), ES {sb*100:.2f}% "
    f"({int(b1.sum())+int(b2.sum())}/{int(m1.sum())+int(m2.sum())}), "
    f"difference {(sb-sa)*100:+.2f} pp")
say(f"- Even pooled over {int(m1.sum())+int(m2.sum())} melanoma cases the test does not reach "
    "p<0.05: every discordant pair falls the same way, but there are only "
    f"{B_+C_} of them, and {B_+C_} unanimous discordances give an exact p of "
    f"{exact2(0, B_+C_):.4f} at best.")

say("\n### How many consistent discordant pairs would be needed\n")
say("| b+c (all favouring ES) | exact two-sided p |")
say("|---|---|")
for nn in range(2, 11):
    say(f"| 0 / {nn} | {exact2(0, nn):.4f} |")
say(f"\nSignificance at alpha=0.05 needs at least {min(nn for nn in range(1,20) if exact2(0,nn)<0.05)} "
    "discordant melanoma cases all falling the same way. We have "
    f"{B_+C_} across {int(m1.sum())+int(m2.sum())} melanoma images, i.e. the current "
    "sample size cannot resolve a difference this small no matter how consistent it is.")

# ---- which melanomas are missed
say("\n### The individual melanoma misses (clean 565)\n")
say("| image | true | ECS pred | ECS p(mel) | ES pred | ES p(mel) | E p(mel) | C p(mel) | S p(mel) |")
say("|---|---|---|---|---|---|---|---|---|")
idx = np.where(m1)[0]
rows = []
for i in idx:
    pe, pc, ps = (PR[k][i] for k in B)
    ecs_p, es_p = ECS[i], ES[i]
    if ecs_p.argmax() != MEL or es_p.argmax() != MEL:
        nm = P(str(pt[i])).name
        say(f"| {nm} | Melanoma | {CLASSES[ecs_p.argmax()]} | {ecs_p[MEL]:.4f} | "
            f"{CLASSES[es_p.argmax()]} | {es_p[MEL]:.4f} | {pe[MEL]:.4f} | {pc[MEL]:.4f} | {ps[MEL]:.4f} |")
        rows.append(dict(image=nm, ecs_pred=CLASSES[ecs_p.argmax()], ecs_p_mel=ecs_p[MEL],
                         es_pred=CLASSES[es_p.argmax()], es_p_mel=es_p[MEL],
                         eff_p_mel=pe[MEL], conv_p_mel=pc[MEL], swin_p_mel=ps[MEL]))
say(f"\nThe entire ECS-vs-ES melanoma gap is {len(rows)} images. ConvNeXt's own melanoma "
    "probability on the cases ECS loses is what drags the three-way average below the "
    "decision boundary -- note that ConvNeXt alone (C) misses 3 melanomas too, the same "
    "count as ECS.")

with open(f"{D}\\weighting_melanoma_misses.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---- the trade: what does ES cost in accuracy, with a CI
say("\n### What switching ECS -> ES would cost elsewhere\n")
for lbl, yy, A, Bb in [("clean test 565", y, ECS, ES), ("validation 748", yv, ECSv, ESv)]:
    pa, pb = A.argmax(1) == yy, Bb.argmax(1) == yy
    bb, cc = int((pa & ~pb).sum()), int((~pa & pb).sum())
    say(f"- {lbl}: accuracy {pa.mean()*100:.2f}% (ECS) vs {pb.mean()*100:.2f}% (ES), "
        f"{(pb.mean()-pa.mean())*100:+.2f} pp; McNemar b={bb}, c={cc}, "
        f"exact two-sided p = {exact2(bb, cc):.4f}")
    # per class recall delta
    say(f"  per-class recall change (ES - ECS):")
    for k, cn in enumerate(CLASSES):
        sel = yy == k
        d = (Bb.argmax(1)[sel] == k).mean() - (A.argmax(1)[sel] == k).mean()
        say(f"    {cn:<22} {d*100:+6.2f} pp   (n={int(sel.sum())})")

say("\n### Macro AUC, the threshold-free comparison\n")


def auc_b(score, pos):
    npos = int(pos.sum())
    nneg = len(pos) - npos
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rr = np.empty(len(s))
    rr[o] = r
    return (rr[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


for lbl, yy, A, Bb in [("clean test 565", y, ECS, ES), ("validation 748", yv, ECSv, ESv)]:
    ma = np.mean([auc_b(A[:, k], yy == k) for k in range(5)])
    mb = np.mean([auc_b(Bb[:, k], yy == k) for k in range(5)])
    say(f"- {lbl}: macro AUC ECS {ma:.5f} vs ES {mb:.5f} ({mb-ma:+.5f}); "
        f"melanoma-vs-rest AUC ECS {auc_b(A[:,MEL], yy==MEL):.5f} vs ES "
        f"{auc_b(Bb[:,MEL], yy==MEL):.5f} ({auc_b(Bb[:,MEL], yy==MEL)-auc_b(A[:,MEL], yy==MEL):+.5f})")
say("\nECS wins macro AUC on both bases. Macro AUC is threshold-free and uses all "
    "565/748 images rather than the 2-3 discordant melanoma cases, so it is the far "
    "better-powered statistic here -- and it points the opposite way to the raw "
    "melanoma-sensitivity ranking.")

say("\n### A cheaper lever than reweighting: the melanoma decision threshold\n")
say("Every melanoma ECS misses is a case where p(melanoma) is high but not the argmax. "
    "Escalating on p(melanoma) >= t instead of requiring argmax recovers them without "
    "touching the weights.\n")
say("| t | ECS mel sensitivity | ECS mel TP | false-positive melanoma flags (non-mel images) | overall acc impact basis |")
say("|---|---|---|---|---|")
nonmel = ~m1
for t in [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    flag = ECS[:, MEL] >= t
    sens_t = (flag[m1]).mean()
    fp = int(flag[nonmel].sum())
    say(f"| {t:.2f} | {sens_t*100:.2f}% | {int(flag[m1].sum())}/{int(m1.sum())} | "
        f"{fp}/{int(nonmel.sum())} ({fp/int(nonmel.sum())*100:.1f}%) | n={len(y)} |")

with open(__file__ + ".out.md",
          "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[body2 written]")
