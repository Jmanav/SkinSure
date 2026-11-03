selftest ok: exact Wilcoxon, sign test and tied ranks all match known values
n = 150 images from the de-duplicated 565-image clean test set (Basal_Cell_Carcinoma 30, Eczema 30, Keratosis 30, Melanoma 30, Tinea 30)

## Overall AOPC (higher = heatmap found pixels the prediction depends on)

| method | mean AOPC | 95% CI (bootstrap) | median |
|---|---|---|---|
| gradcam | 0.3099 | [0.2831, 0.3376] | 0.2720 |
| random7x7 | 0.1834 | [0.1678, 0.1994] | 0.1612 |
| randompixel | 0.4887 | [0.4576, 0.5196] | 0.4837 |

## Paired comparisons (Grad-CAM minus baseline, same images)

| comparison | mean diff | 95% CI | Wilcoxon W | p | sign test | p |
|---|---|---|---|---|---|---|
| gradcam - random7x7 | 0.1266 | [0.1059, 0.1478] | 747.0000 (n=150) | <0.0001 | 127/150 positive | <0.0001 |
| gradcam - randompixel | -0.1788 | [-0.2236, -0.1345] | 2190.0000 (n=150) | <0.0001 | 44/150 positive | <0.0001 |

**`random7x7` is the fair comparator; `randompixel` is a sanity check, not a rival.** Both baselines delete the same NUMBER of pixels, but not the same KIND of region. `random7x7` draws noise on the 7x7 Grad-CAM grid and upsamples it, so its deletions are contiguous blobs matched in spatial structure to a Grad-CAM map -- the only difference is whether the blobs are in informative places. `randompixel` scatters i.i.d. noise at full 224x224 resolution, so it removes a fine-grained speckle that shreds local texture everywhere at once and pushes the image far off the training distribution. That collapses p(class) no matter what it covers, which is why it posts the highest AOPC while explaining nothing. A high `randompixel` AOPC therefore measures the models' fragility to high-frequency corruption, NOT localisation quality; read `gradcam - random7x7` as the result and treat `gradcam - randompixel` as confirmation that AOPC is sensitive to out-of-distribution damage and so must be compared only against a structure-matched baseline.

## Per-class AOPC (per-class n is in the table; intervals are wide and each row is individually underpowered)

| true class | n | gradcam | random7x7 | randompixel | gradcam-random7x7 | Wilcoxon p |
|---|---|---|---|---|---|---|
| Basal_Cell_Carcinoma | 30 | 0.2667 [0.2189,0.3161] | 0.1907 [0.1661,0.2174] | 0.6576 [0.6182,0.6972] | 0.0760 | 0.0012 |
| Eczema | 30 | 0.3336 [0.2842,0.3853] | 0.2247 [0.1854,0.2649] | 0.4382 [0.3677,0.5073] | 0.1089 | <0.0001 |
| Keratosis | 30 | 0.3538 [0.2908,0.4193] | 0.1697 [0.1335,0.2054] | 0.3905 [0.3368,0.4435] | 0.1841 | <0.0001 |
| Melanoma | 30 | 0.1918 [0.1450,0.2449] | 0.1133 [0.0961,0.1333] | 0.5889 [0.5278,0.6468] | 0.0786 | <0.0001 |
| Tinea | 30 | 0.4037 [0.3476,0.4628] | 0.2184 [0.1851,0.2551] | 0.3683 [0.3221,0.4152] | 0.1853 | <0.0001 |

## Split by whether the ensemble's prediction was correct

| subset | n | gradcam | random7x7 | randompixel | gradcam-random7x7 | Wilcoxon p |
|---|---|---|---|---|---|---|
| correct | 128 | 0.2966 | 0.1737 | 0.5113 | 0.1229 | <0.0001 |
| incorrect | 22 | 0.3876 | 0.2397 | 0.3571 | 0.1479 | 0.0006 |

## Mean deletion curve: p(predicted class) as the top-k pixels are blanked

| deleted % | gradcam | random7x7 | randompixel |
|---|---|---|---|
| 0% | 0.8685 | 0.8685 | 0.8685 |
| 10% | 0.7773 | 0.8508 | 0.6630 |
| 20% | 0.7169 | 0.8415 | 0.5379 |
| 30% | 0.6709 | 0.8231 | 0.4617 |
| 40% | 0.6249 | 0.8062 | 0.4079 |
| 50% | 0.5914 | 0.7828 | 0.3642 |
| 60% | 0.5680 | 0.7410 | 0.3230 |
| 70% | 0.5333 | 0.6997 | 0.3019 |
| 80% | 0.4923 | 0.6190 | 0.2792 |
| 90% | 0.4067 | 0.4833 | 0.2552 |
| 100% | 0.2043 | 0.2043 | 0.2043 |

## Effect size and win rate (vs random7x7)

- Grad-CAM scored higher on **127/150** images (84.7%).
- Mean paired difference 0.1266, SD 0.1309, Cohen's dz = 0.967.
- Matched-pairs rank-biserial correlation +0.693.

Grad-CAM build + deletion curve: mean 0.05 s/image, median 0.04, max 0.20.
