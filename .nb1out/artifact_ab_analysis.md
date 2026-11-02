loaded 565 paired rows from F:\Projects\SkinSure\.nb1out\artifact_ab_raw.csv

==============================================================================
1. remove_artifacts FIRING RATE
==============================================================================
  overall applied=True  335/565 = 0.5929  95% CI [0.5519, 0.6327]
  overall skipped       230/565 = 0.4071

  class                     n  fired     rate                95% CI
  Basal_Cell_Carcinoma    106     46   0.4340   [0.3436, 0.5290]
  Eczema                  111     75   0.6757   [0.5840, 0.7556]
  Keratosis               133     87   0.6541   [0.5700, 0.7296]
  Melanoma                124     55   0.4435   [0.3591, 0.5314]
  Tinea                    91     72   0.7912   [0.6968, 0.8621]

  reason the step declined, by bucket:
    APPLIED (inpainted)                         335  (59.29%)
    no artifacts detected (mask empty)          206  (36.46%)
    skipped: mask > 10% (guard tripped)          24  ( 4.25%)

==============================================================================
2. WHOLE SET -- A (baseline) vs B (artifact removal attempted on all)  (n=565)
==============================================================================

accuracy   A(baseline)   0.8761  [0.8464, 0.9008]  495/565
           B(artifact)   0.8779  [0.8483, 0.9024]  496/565
           delta B-A    +0.0018  (+0.18 pp)

macro AUC (one-vs-rest)   A 0.9813   B 0.9813   delta -0.0001
  class                     AUC A    AUC B    delta
  Basal_Cell_Carcinoma     0.9970   0.9970  -0.0000
  Eczema                   0.9707   0.9703  -0.0004
  Keratosis                0.9665   0.9670  +0.0005
  Melanoma                 0.9987   0.9988  +0.0001
  Tinea                    0.9737   0.9732  -0.0005

per-class sensitivity / recall
  class                     n            sens A            sens B    delta
  Basal_Cell_Carcinoma    106  1.0000 [0.965,1.000]  1.0000 [0.965,1.000]  +0.0000  <<<
  Eczema                  111  0.9009 [0.831,0.944]  0.9009 [0.831,0.944]  +0.0000
  Keratosis               133  0.7068 [0.624,0.777]  0.7068 [0.624,0.777]  +0.0000
  Melanoma                124  0.9758 [0.931,0.992]  0.9758 [0.931,0.992]  +0.0000  <<<
  Tinea                    91  0.8132 [0.721,0.880]  0.8242 [0.733,0.889]  +0.0110

SAFETY-CRITICAL (Wilson 95% CI)
  Melanoma sensitivity  n=124
    A 121/124 = 0.9758   95% CI [0.9313, 0.9917]
    B 121/124 = 0.9758   95% CI [0.9313, 0.9917]
    paired McNemar on this class: b=0 c=0  exact p=1.0000
  Basal_Cell_Carcinoma sensitivity  n=106
    A 106/106 = 1.0000   95% CI [0.9650, 1.0000]
    B 106/106 = 1.0000   95% CI [0.9650, 1.0000]
    paired McNemar on this class: b=0 c=0  exact p=1.0000

McNemar's test (paired, same images, A vs B)
  both correct                     495
  both wrong                       69
  b = A correct, B wrong           0   (artifact removal broke)
  c = A wrong,   B correct         1   (artifact removal fixed)
  discordant pairs n = b+c         1
  chi2 (Yates-corrected, 1 df)     0.0000   p = 1.0000
  exact binomial b vs b+c          p = 1.0000

label flips A->B (any direction): 1/565 (0.18%)
max per-class prob shift: mean 0.00478  median 0.00020  p95 0.02748  max 0.12134

==============================================================================
3. FIRED SUBSET ONLY -- images where remove_artifacts reported applied=True  (n=335)
==============================================================================

accuracy   A(baseline)   0.8567  [0.8151, 0.8902]  287/335
           B(artifact)   0.8597  [0.8184, 0.8928]  288/335
           delta B-A    +0.0030  (+0.30 pp)

macro AUC (one-vs-rest)   A 0.9775   B 0.9774   delta -0.0001
  class                     AUC A    AUC B    delta
  Basal_Cell_Carcinoma     0.9964   0.9963  -0.0001
  Eczema                   0.9640   0.9633  -0.0007
  Keratosis                0.9596   0.9603  +0.0007
  Melanoma                 0.9979   0.9980  +0.0001
  Tinea                    0.9696   0.9689  -0.0006

per-class sensitivity / recall
  class                     n            sens A            sens B    delta
  Basal_Cell_Carcinoma     46  1.0000 [0.923,1.000]  1.0000 [0.923,1.000]  +0.0000  <<<
  Eczema                   75  0.8667 [0.772,0.926]  0.8667 [0.772,0.926]  +0.0000
  Keratosis                87  0.7241 [0.622,0.807]  0.7241 [0.622,0.807]  +0.0000
  Melanoma                 55  0.9455 [0.851,0.981]  0.9455 [0.851,0.981]  +0.0000  <<<
  Tinea                    72  0.8472 [0.747,0.912]  0.8611 [0.763,0.923]  +0.0139

SAFETY-CRITICAL (Wilson 95% CI)
  Melanoma sensitivity  n=55
    A 52/55 = 0.9455   95% CI [0.8515, 0.9813]
    B 52/55 = 0.9455   95% CI [0.8515, 0.9813]
    paired McNemar on this class: b=0 c=0  exact p=1.0000
  Basal_Cell_Carcinoma sensitivity  n=46
    A 46/46 = 1.0000   95% CI [0.9229, 1.0000]
    B 46/46 = 1.0000   95% CI [0.9229, 1.0000]
    paired McNemar on this class: b=0 c=0  exact p=1.0000

McNemar's test (paired, same images, A vs B)
  both correct                     287
  both wrong                       47
  b = A correct, B wrong           0   (artifact removal broke)
  c = A wrong,   B correct         1   (artifact removal fixed)
  discordant pairs n = b+c         1
  chi2 (Yates-corrected, 1 df)     0.0000   p = 1.0000
  exact binomial b vs b+c          p = 1.0000

label flips A->B (any direction): 1/335 (0.30%)
max per-class prob shift: mean 0.00806  median 0.00181  p95 0.04125  max 0.12134

==============================================================================
4. SKIPPED SUBSET (control: B must be identical to A here)  (n=230)
==============================================================================

accuracy   A(baseline)   0.9043  [0.8594, 0.9360]  208/230
           B(artifact)   0.9043  [0.8594, 0.9360]  208/230
           delta B-A    +0.0000  (+0.00 pp)

macro AUC (one-vs-rest)   A 0.9860   B 0.9860   delta +0.0000
  class                     AUC A    AUC B    delta
  Basal_Cell_Carcinoma     0.9972   0.9972  +0.0000
  Eczema                   0.9808   0.9808  +0.0000
  Keratosis                0.9761   0.9761  +0.0000
  Melanoma                 1.0000   1.0000  +0.0000
  Tinea                    0.9758   0.9758  +0.0000

per-class sensitivity / recall
  class                     n            sens A            sens B    delta
  Basal_Cell_Carcinoma     60  1.0000 [0.940,1.000]  1.0000 [0.940,1.000]  +0.0000  <<<
  Eczema                   36  0.9722 [0.858,0.995]  0.9722 [0.858,0.995]  +0.0000
  Keratosis                46  0.6739 [0.530,0.791]  0.6739 [0.530,0.791]  +0.0000
  Melanoma                 69  1.0000 [0.947,1.000]  1.0000 [0.947,1.000]  +0.0000  <<<
  Tinea                    19  0.6842 [0.460,0.846]  0.6842 [0.460,0.846]  +0.0000

SAFETY-CRITICAL (Wilson 95% CI)
  Melanoma sensitivity  n=69
    A 69/69 = 1.0000   95% CI [0.9473, 1.0000]
    B 69/69 = 1.0000   95% CI [0.9473, 1.0000]
    paired McNemar on this class: b=0 c=0  exact p=1.0000
  Basal_Cell_Carcinoma sensitivity  n=60
    A 60/60 = 1.0000   95% CI [0.9398, 1.0000]
    B 60/60 = 1.0000   95% CI [0.9398, 1.0000]
    paired McNemar on this class: b=0 c=0  exact p=1.0000

McNemar's test (paired, same images, A vs B)
  both correct                     208
  both wrong                       22
  b = A correct, B wrong           0   (artifact removal broke)
  c = A wrong,   B correct         0   (artifact removal fixed)
  discordant pairs n = b+c         0
  chi2 (Yates-corrected, 1 df)     nan   p = 1.0000
  exact binomial b vs b+c          p = 1.0000

label flips A->B (any direction): 0/230 (0.00%)
max per-class prob shift: mean 0.00000  median 0.00000  p95 0.00000  max 0.00000

==============================================================================
5. CONTROL CHECK -- determinism on non-fired images
==============================================================================
  n skipped = 230
  max absolute probability difference over all classes: 0.00000000
  images with ANY nonzero difference: 0
  label flips among skipped images: 0
  -> 0.00000000 means the scoring path is deterministic and every flip observed above is a genuine inpainting effect.

==============================================================================
6. EVERY LABEL FLIP, LISTED
==============================================================================

  v-tinea-face-4  true=Tinea
    A: Keratosis              p=0.4655      (p_true=0.4566)
    B: Tinea                  p=0.4882      (p_true=0.4882)
    mask: combined 0.3630% (ink 0.0003%, glare 0.3627%, ruler 0.0000%)
    note: inpainted 0.4% of pixels
    -> FIXED (was wrong, now right)

==============================================================================
7. MASK COVERAGE DIAGNOSTICS
==============================================================================

  combined mask fraction of frame:
  ALL images                         n= 565  min 0.0000%  p25 0.0000%  median 0.0425%  p75 0.9889%  max 47.9132%
  FIRED (inpainted)                  n= 335  min 0.0002%  p25 0.0540%  median 0.3069%  p75 1.5913%  max 9.9920%
  SKIPPED (any reason)               n= 230  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0000%  max 47.9132%
  SKIPPED by >10% guard              n=  24  min 10.0147%  p25 11.3319%  median 14.7138%  p75 19.7642%  max 47.9132%
  SKIPPED mask empty                 n= 206  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0000%  max 0.0000%

  per-detector contribution (fraction of frame), ALL images:
  ink                                n= 565  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0000%  max 47.8458%
  glare                              n= 565  min 0.0000%  p25 0.0000%  median 0.0084%  p75 0.3619%  max 14.7315%
  ruler                              n= 565  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0000%  max 18.8423%

  per-detector, FIRED images only:
  ink                                n= 335  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0027%  max 9.9358%
  glare                              n= 335  min 0.0000%  p25 0.0139%  median 0.1694%  p75 0.9850%  max 9.5810%
  ruler                              n= 335  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.0000%  max 5.2196%

  per-detector, guard-tripped images only:
  ink                                n=  24  min 0.0000%  p25 0.0000%  median 4.2700%  p75 17.1849%  max 47.8458%
  glare                              n=  24  min 0.0000%  p25 0.0937%  median 2.2403%  p75 10.1495%  max 14.7315%
  ruler                              n=  24  min 0.0000%  p25 0.0000%  median 0.0000%  p75 0.6781%  max 18.8423%

  detector that dominates each image's mask (argmax of the three):
    all             ink=69  glare=260  ruler=30  no mask at all=206
    fired           ink=57  glare=252  ruler=26  no mask at all=0
    guard-tripped   ink=12  glare=8  ruler=4  no mask at all=0

  what the >10% guard is suppressing:
    images the guard stopped from being inpainted: 24 (4.25% of 565)
    their mask coverage: median 14.71%, mean 19.83%, max 47.91%
      >= 25% of the frame: 5 images
      >= 50% of the frame: 0 images
      >= 75% of the frame: 0 images
    without the guard, 359 of 565 images (63.5%) would be inpainted.

    guard trips by class:
      Basal_Cell_Carcinoma     0/106  = 0.00%   median mask nan%
      Eczema                  11/111  = 9.91%   median mask 14.73%
      Keratosis                4/133  = 3.01%   median mask 13.58%
      Melanoma                 2/124  = 1.61%   median mask 12.50%
      Tinea                    7/91   = 7.69%   median mask 16.91%

  firing rate vs mask size, for context on how close the fired images sit to the cutoff:
    fired-image masks as a share of the 10% cutoff: median 0.03x, p90 0.35x
    fired images with mask < 1% of frame: 220/335
