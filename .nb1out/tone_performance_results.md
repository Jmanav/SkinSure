# Per-Fitzpatrick/ITA Tone-Bin Accuracy — NB3 (skin_tone_ita)

**Status: PRELIMINARY — not yet in `main_updated.tex` as a validated result pending
re-run on the de-duplicated test set. Saved here as the raw record of the run that
produced `fig6_tone_performance.pdf` and `tone_results.json` on Kaggle.**

Source notebook: `paper/experiments/03_skin_tone_ita.ipynb`
Run date: 2026-11-05 (Kaggle)

## Important caveat — test set

**This run used the full 727-image contaminated test set, not the de-duplicated
565-image set used everywhere else in the paper.** Notebook 2's pHash audit found
162 of 727 test images (22.3%) are near-duplicates of train/val images. Every other
reported number in `main_updated.tex` is on the clean 565 set for exactly this
reason. These tone-bin numbers have **not** been re-run on the clean subset and are
not directly comparable to the paper's other accuracy figures until that re-run
happens.

## Method

- `tone_bin` assigned from individual typology angle (ITA) computed per image
  (method in `03_skin_tone_ita.ipynb`), six bins: very light, light, intermediate,
  tan, brown, dark.
- Accuracy per bin, per backbone and for the equal-weight ensemble, with a
  2,000-resample bootstrap 95% CI (seed consistent with the rest of NB1/NB2/NB3).
- Bins with n < 30 are flagged `unreliable_n_lt_30` (none triggered this run — all
  six bins had n >= 92).

## Results (n=727, full contaminated test set)

### EfficientNet-B1

| tone bin | n | accuracy | 95% CI |
|---|---|---|---|
| very light | 219 | 89.50% | [85.39, 93.15] |
| light | 100 | 80.00% | [72.00, 88.00] |
| intermediate | 93 | 88.17% | [81.72, 94.62] |
| tan | 103 | 80.58% | [72.82, 88.35] |
| brown | 92 | 83.70% | [76.09, 91.30] |
| dark | 120 | 81.67% | [74.17, 88.33] |

### ConvNeXt-Tiny

| tone bin | n | accuracy | 95% CI |
|---|---|---|---|
| very light | 219 | 92.69% | [89.04, 95.89] |
| light | 100 | 87.00% | [80.00, 93.00] |
| intermediate | 93 | 93.55% | [88.17, 97.85] |
| tan | 103 | 87.38% | [80.58, 93.20] |
| brown | 92 | 83.70% | [76.09, 91.30] |
| dark | 120 | 86.67% | [80.83, 92.50] |

### Swin-Tiny

| tone bin | n | accuracy | 95% CI |
|---|---|---|---|
| very light | 219 | 89.95% | [85.84, 93.61] |
| light | 100 | 84.00% | [76.98, 91.00] |
| intermediate | 93 | 88.17% | [81.72, 94.62] |
| tan | 103 | 84.47% | [76.70, 91.26] |
| brown | 92 | 80.43% | [71.74, 88.04] |
| dark | 120 | 84.17% | [77.50, 90.83] |

### Ensemble (E+C+S)

| tone bin | n | accuracy | 95% CI |
|---|---|---|---|
| very light | 219 | 92.24% | [88.58, 95.43] |
| light | 100 | 85.00% | [78.00, 92.00] |
| intermediate | 93 | 94.62% | [90.32, 98.92] |
| tan | 103 | 85.44% | [78.64, 92.23] |
| brown | 92 | 82.61% | [75.00, 89.16] |
| dark | 120 | 85.83% | [79.17, 91.67] |

**Ensemble tone gap (all bins reliable, n >= 92 each):** best bin "intermediate"
94.62% vs worst bin "brown" 82.61% — a gap of 12.01 pp. The 95% CIs of these two
bins do not overlap (intermediate [90.32, 98.92] vs brown [75.00, 89.16]), so on
this (contaminated) test set the gap is not attributable to sampling noise alone —
but see caveats below before treating this as a validated fairness finding.

## What this does and does not show

- The ensemble's best- and worst-performing tone bins do not fall at the extremes
  of the ITA scale: "very light" (n=219, the largest bin, dominated by Western
  dermoscopic sources) scores 92.24%, close to the top, while "brown" — not "dark"
  — is the worst at 82.61%. A naive "lighter skin always wins" story is not what
  the data shows; the gap does not monotonically track lightness.
- All four models (three backbones + ensemble) show the same qualitative pattern:
  "very light" and "intermediate" bins score highest, "tan" and "brown" score
  lowest, "dark" sits in between. This consistency across independently trained
  architectures suggests the gap tracks something about the **data** (e.g. which
  source corpus and which disease classes dominate each tone bin) rather than being
  an architecture-specific artefact.
- This is **not yet a controlled fairness measurement**: tone bin is confounded
  with disease-class mix and source dataset, and that confound has not been
  decomposed here. A bin's accuracy reflects both its skin-tone difficulty and
  which diseases happen to be overrepresented in it.

## Outstanding before this can be reported as a result

1. **Re-run on the de-duplicated 565-image test set.** This is the blocking item —
   every other number in the paper uses this set specifically because the full 727
   is contaminated by near-duplicate leakage.
2. **Decompose the tone/class confound.** Report per-class accuracy within each
   tone bin, or control for class mix, before attributing the gap to skin tone
   rather than to which diseases are overrepresented in which bin.
3. **Sanity-check the ITA-to-bin mapping** against a small manually-labelled sample,
   since "brown" scoring worse than "dark" is not the pattern a reader will expect
   and deserves a defensible explanation beyond the raw number.

Until (1)-(3) are done, treat this file as a preliminary record of a real run, not
as a reportable finding. `main_updated.tex` Section 8 (Future Scope) describes this
work as in-progress with these caveats stated explicitly; it is not used anywhere
in the Results section.
