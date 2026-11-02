"""The agreement tiers NB1 reported are mis-binned.

NB1 compares a float agreement score against the literals 0.33 and 0.67, but
the score can only be 0, 1/3, 2/3 or 1. 1/3 = 0.3333... != 0.33, so the two
mixed tiers match nothing and report n=0.
"""
import collections
import csv

import numpy as np
from pathlib import PurePosixPath as P

d = r"F:\Projects\SkinSure\.nb1out"
MAN = r"C:\Users\Cipher\Downloads\test_clean_manifest.csv"

Z = {k: np.load(f"{d}\cache_{k}_test.npz", allow_pickle=True)
     for k in ["efficientnet_b1", "convnext_tiny", "swin_tiny"]}
paths, y0 = Z["efficientnet_b1"]["paths"], Z["efficientnet_b1"]["labels"]
keep = {f"{P(r['path']).parent.name}/{P(r['path']).name}"
        for r in csv.DictReader(open(MAN, encoding="utf-8-sig"))}
mask = np.array([f"{P(str(p)).parent.name}/{P(str(p)).name}" in keep for p in paths])

y = y0[mask]
E, C, S = [Z[k]["probs"][mask] for k in
           ["efficientnet_b1", "convnext_tiny", "swin_tiny"]]
ens = ((E + C + S) / 3).argmax(1)
tops = np.stack([E.argmax(1), C.argmax(1), S.argmax(1)], 1)
agree = (tops == ens[:, None]).sum(1) / 3
correct = ens == y

print("actual agreement tiers on the clean 565:")
tot = 0
for v, n in sorted(collections.Counter(np.round(agree, 4)).items()):
    sel = np.isclose(agree, v)
    tot += n
    print(f"  {v:.4f} ({int(round(v*3))}/3)  n={n:>4}  "
          f"pct={100*n/len(y):5.2f}%  ensemble acc={correct[sel].mean():.4f}")
print(f"  total n={tot} of {len(y)}")

print("\nwhat NB1 reported: 0.33 -> n=0, 0.67 -> n=0, 1.0 -> n=484")
print("the two mixed tiers hold 81 images (14.3%) that were dropped from the table.")

# The headline split NB1 did get right, because it used a different code path.
un = np.isclose(agree, 1.0)
print(f"\nunanimous      n={un.sum():>4}  acc={correct[un].mean():.4f}")
print(f"not unanimous  n={(~un).sum():>4}  acc={correct[~un].mean():.4f}")

# Triage relevance: confidence_tier() maps these to high/moderate/low.
print("\nper CLAUDE.md, Predictions.confidence_tier() maps:")
print("  3/3 -> high, 2/3 -> moderate, otherwise low")
for lab, sel in [("high (3/3)", np.isclose(agree, 1.0)),
                 ("moderate (2/3)", np.isclose(agree, 2 / 3)),
                 ("low (<=1/3)", agree <= 1 / 3 + 1e-9)]:
    if sel.sum():
        print(f"  {lab:16s} n={sel.sum():>4}  acc={correct[sel].mean():.4f}")
