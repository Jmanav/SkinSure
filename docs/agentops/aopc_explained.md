# AOPC: how we test whether a Grad-CAM heatmap is telling the truth

Written for the SkinSure xAI faithfulness experiment. Code referenced here:
`src/explainability/evaluation.py`, `scripts/xai_aopc_experiment.py`,
`scripts/xai_aopc_analyze.py`.

---

## 1. The problem: a heatmap always looks plausible

Run Grad-CAM on any lesion photo and you get a warm blob somewhere on the image. It
looks convincing. That is exactly the danger.

A heatmap is a *picture of a claim*: "the model's prediction depends on these pixels."
Nothing in the act of drawing the picture verifies that claim. Grad-CAM is gradients times
activations at one chosen layer — a defensible recipe, but still a recipe. If the recipe
were subtly wrong (wrong target layer, wrong channel weighting, Swin's NHWC axes
transposed the wrong way), you would still get a smooth, centred, entirely
plausible-looking blob. Your eyes cannot tell those apart.

So you need a test that does not consult your eyes. That property — "does the explanation
actually identify the pixels the prediction relies on?" — is **explanation faithfulness**.
It is not the same as looking right, and not the same as agreeing with a dermatologist. It
is purely about the model: is the explanation a true statement about this network's
behaviour?

## 2. The deletion curve

The idea is blunt and that is its strength. If a heatmap really has found the pixels the
prediction depends on, then **destroying those pixels should destroy the prediction.**

Procedure, implemented in `deletion_curve()` in `src/explainability/evaluation.py`:

1. Flatten the 224x224 heatmap and sort every pixel by `|heatmap|`, descending.
2. For each level in `DELETION_PERCENTS = (0, 10, 20, ..., 100)`, blank the top-k% most
   important pixels.
3. Record p(predicted class) at each level.

Two details worth knowing:

- **"Blank" means write `0.0` in normalised space**, which is the ImageNet mean colour,
  not black. Black is a strong, unusual signal; the mean colour is the least informative
  value available in the tensor's own coordinate system.
- **The class explained is the ensemble's argmax, not the ground truth.** Faithfulness
  asks "did the map find the pixels *this prediction* used," so when the model is wrong
  we still explain the wrong answer. That is deliberate.

You now have 11 numbers. A faithful heatmap gives a curve that falls fast — the first few
deletions take out the pixels that mattered. An unfaithful one stays flat until you have
deleted so much of the image that anything would break.

### AOPC collapses the curve into one number

**AOPC** = Area Over the Perturbation Curve. In this repo (`aopc()`, same file) it is
the mean drop from the undeleted confidence:

```python
def aopc(curve):
    return float(np.mean([curve[0] - c for c in curve[1:]]))
```

`curve[0]` is the original confidence; each later entry is the confidence after deleting
more. AOPC averages all ten drops. **Higher AOPC = more faithful.** A heatmap that points
at nothing keeps confidence high, so the drops are small, so AOPC is small.

Note the 100% level is included, where every method has deleted the whole image and so
records an identical value. It cannot create a difference between methods — it just drags
every AOPC up by the same constant.

## 3. A baseline is not optional

Here is a real row from `.nb1out/xai_aopc_raw.csv` — one BCC image, Grad-CAM:

```
AOPC = 0.147
```

Is 0.147 good? **The question is unanswerable.** AOPC has no natural scale — it depends on
the starting confidence, how robust the backbones are to occlusion, the lesion's size, how
much of the frame is plain skin. A bare number says nothing about localisation.

The fix is comparative: score a *random* heatmap through the identical pipeline and ask
whether the real method beats it. Hence the governing principle:

> **A method that cannot beat random explains nothing.**

If scribbling noise on the image and deleting wherever it happens to be bright knocks the
confidence down just as hard as your carefully-computed Grad-CAM map, then your map
carries no localisation information. It may still be a pretty picture. It is not an
explanation.

This is why `scripts/evaluate_xai.py` and `scripts/xai_aopc_experiment.py` both score
Grad-CAM, external heatmaps and a random baseline **under one identical procedure**. Same
deletion levels, same blanking value, same ordering code, same ensemble. Any difference
in the numbers is then attributable to the heatmaps and nothing else.

## 4. The subtlety that matters most: two random baselines, not interchangeable

This is the part that is easy to get wrong, and getting it wrong inverts the conclusion.

There are two obvious ways to build a "random heatmap," and they are **not** two attempts
at the same thing. They measure different phenomena.

### `random7x7` — the fair comparator

Draw uniform noise on a **7x7 grid**, then bilinearly upsample it to 224x224 using the
same `F.interpolate(..., mode="bilinear", align_corners=False)` call Grad-CAM uses.

7x7 is not arbitrary: it is Grad-CAM's **native resolution** in all three backbones at
224px input (`CAM_GRID = 7` in the experiment script). Grad-CAM genuinely only has 49
spatial cells of information; the 224x224 map you look at is an upsampling artifact.

So `random7x7` produces maps with the *same spatial structure* as Grad-CAM: smooth,
blobby, low-frequency. When you delete the top-30%, you remove a few **contiguous
regions** of comparable size in both cases. The deletions look the same kind of damage.

**The only difference between Grad-CAM and `random7x7` is whether the blobs land on
informative pixels.** That is precisely the question faithfulness asks. This is the
comparator the result rests on.

### `randompixel` — a sanity check, not a rival

Draw i.i.d. uniform noise at full **224x224** resolution. Every pixel is independent.

Deleting the top-30% of that map removes a **fine speckle scattered across the entire
frame**. Nothing is spatially coherent. This does two things that have nothing to do with
localisation:

- It **shreds local texture everywhere at once**. Texture is what a CNN's early layers
  read. A salt-and-pepper mask destroys texture globally, including over the lesion.
- It pushes the image **far off the training distribution**. The models never saw
  speckled images. They were trained on resize + ImageNet normalisation, full stop.

Result: p(class) collapses *regardless of where the deletions are*, because the input is
no longer the kind of thing the network knows how to read.

### The punchline

Concrete, from the cached rows for `ISIC_0058222.jpg` (BCC, predicted BCC at 0.940):

| method | p after 10% deleted | p after 20% | AOPC |
|---|---|---|---|
| gradcam | 0.936 | 0.894 | **0.147** |
| random7x7 | 0.932 | 0.934 | 0.187 |
| randompixel | 0.433 | 0.157 | **0.797** |

**`randompixel` posts by far the highest AOPC while explaining absolutely nothing.** Its
"heatmap" is noise; it has zero localisation content by construction. It wins because it
is the most destructive perturbation, not because it found anything.

The lesson generalises beyond this repo:

> A high AOPC can be earned two different ways — by locating important pixels, or by
> damaging the image in a way the model cannot cope with. AOPC does not distinguish
> these. Only a **structure-matched** baseline does.

So the rule for reading this experiment: **`gradcam - random7x7` is the result.
`gradcam - randompixel` is confirmation that AOPC is sensitive to out-of-distribution
damage, which is exactly why a structure-matched baseline is mandatory.** Never quote
`randompixel` as a method that beat Grad-CAM. The explanatory paragraph in
`scripts/xai_aopc_analyze.py` (search `fair comparator`) says the same thing and is the
canonical wording.

## 5. Reading this project's output

`scripts/xai_aopc_analyze.py` reads the cached CSV — no model reloading, so re-running the
analysis is free — and prints several tables.

**Sample.** 150 images: 30 per class x 5 classes, stratified, drawn with a fixed seed from
the de-duplicated **565-image** clean test set. (SkinSure has three test sets in play —
997 / 727 / 565 — which must never be compared with each other. This is the 565.)

**Seeds.** Each random baseline is drawn 3 times per image with seeds derived from the
filename, so the baseline is not one lucky draw. `load()` averages a method's seeds first,
giving exactly **one AOPC per (image, method)** before any statistics run.

### Why the comparison is paired

Every image is scored by *every* method. So for image *i* you have both
`gradcam_i` and `random7x7_i`, and you can form the per-image difference
`d_i = gradcam_i - random7x7_i`.

This matters enormously. AOPC varies hugely *between* images — a high-confidence melanoma
on clear skin behaves nothing like a low-confidence keratosis. If you compared two
independent groups of images, that between-image variance would swamp the effect you care
about. Pairing cancels it: each image acts as its own control, and you only ask "within
this image, did Grad-CAM beat noise?" Far more statistical power from the same 150 images.

### The tests

**Wilcoxon signed-rank (the headline test).** Ranks the per-image differences `d_i` by
`|d_i|` and sums the positive ranks against the negative ones. Under the null ("Grad-CAM
and random7x7 are equivalent") each sign is a coin flip, so the two sums should be similar.
`W` is the smaller sum; a small `W` means the differences pile up on one side. It uses
ranks, not raw values, so one freak image cannot drive the result — important where AOPC is
bounded and skewed. scipy is absent from this venv, so `wilcoxon()` is implemented in the
analysis script (exact enumeration for n <= 20, otherwise normal approximation with tie and
continuity correction) and `_selftest()` checks it against textbook cases (`--selftest`).

**Sign test (distribution-free check).** Discards magnitudes and just counts: of the
nonzero differences, how many favour Grad-CAM? Compared to a binomial coin flip. Weaker
than Wilcoxon but assumes almost nothing — if the two disagree, Wilcoxon was leaning on its
assumptions.

**Win rate.** Grad-CAM scored higher on `k/150` images — the most readable number in the
report.

**Cohen's dz.** `mean(d) / sd(d)`, the paired effect size. The p-value says "probably not
zero"; dz says **how big** (~0.2 small, ~0.5 medium, ~0.8 large). A tiny dz with a
significant p just means n sufficed to detect something trivial. Both are needed.

**Matched-pairs rank-biserial correlation.** `(wins - losses) / nonzero`, in [-1, +1] — how
lopsided the signs are. **Bootstrap 95% CIs** resample images 20,000 times and take the
2.5th/97.5th percentiles of the mean, so no normality is assumed.

**Per-class table.** Each row has n = 30, so intervals are wide and every row is
individually underpowered — the script says so in its heading. Per-class rows flag
directions worth investigating, not per-class effects.

**The PARTIAL RUN warning.** The analysis script prints a loud warning when fewer than 5
classes are present, because **the sample manifest is ordered by class**. An interrupted
run is therefore not a random subset — it is all of BCC, then all of Eczema, and so on.
A partial run is **class-biased and must never be quoted as a finding**, only as "classes
X and Y have been scored so far."

## 6. Why this matters for SkinSure

**Gap 10 — Grad-CAM's place on `/predict` has never been earned on faithfulness.**
CLAUDE.md is explicit: Grad-CAM sits on the live path for **latency** (one forward + one
backward per backbone, ~5 s on the dev CPU, versus minutes for SHAP's hundreds-to-thousands
of forward passes) and **output shape** (a dense 7x7 spatial map that reduces naturally to
`XAISummary`; SHAP gives per-patch attributions needing their own conversion first). Those
are real engineering reasons. Neither is evidence that Grad-CAM *explains better*.

This experiment is the **first measurement** of that caveat. The cheapest useful version —
Grad-CAM vs random, no SHAP, no API spend — is the right first step: if Grad-CAM cannot
beat a structure-matched noise baseline, there is nothing to compare SHAP *to*, and the
integration question answers itself.

**Gap 7 — the heatmap -> `XAISummary` bridge is heuristic and unvalidated.** The bridge
turns the heatmap into `focus_description` and `boundary_alignment` using
first-guess thresholds and a brightness-based lesion segmentation, not a trained
segmenter. On the 15 real-photo run, `boundary_alignment` came back `poor` **9 of 15 times
and `good` never once**, and `focus_description` was `background_skin` on 4 of 15 —
including two *correct, high-confidence* melanoma calls (0.957, 0.853).

AOPC speaks directly to the ambiguity there. Those gap-7 observations have two rival
explanations:

- **(a)** The heatmaps are fine; the bridge's thresholds and segmentation are mis-tuned.
- **(b)** The heatmaps genuinely do not track lesions, so the bridge is correctly
  reporting a real problem upstream.

A solid `gradcam - random7x7` win says the heatmaps carry real localisation information
and pushes the blame toward **(a)** — a tuning job against annotated masks. A null result
pushes it toward **(b)**, which would be a much more serious finding, because the artifact
flag forces a re-capture and a false flag costs a patient a retake.

Note AOPC cannot fully settle gap 7 on its own: it asks whether the map found pixels the
*model* used, not whether those pixels are *the lesion*. If the models genuinely exploit
perilesional skin, a faithful map will point at background skin and be right to. Settling
that needs annotated lesion masks. AOPC narrows the question; it does not close it.

---

## Quick reference

| Thing | Where |
|---|---|
| `DELETION_PERCENTS`, `deletion_curve()`, `aopc()`, `ensemble_probs_fn()` | `src/explainability/evaluation.py` |
| Run the experiment (local torch, no API spend) | `scripts/xai_aopc_experiment.py` |
| Analyse the cached rows (free to re-run) | `scripts/xai_aopc_analyze.py` |
| Raw per-(image, method, seed) rows | `.nb1out/xai_aopc_raw.csv` |
| Saved report | `.nb1out/xai_aopc_analysis.md` |
| Grad-CAM itself, per-backbone target layers | `src/explainability/gradcam.py` |

```bash
# resume a run (safe to interrupt; --resume re-scores the trailing partial image)
./venv/Scripts/python.exe scripts/xai_aopc_experiment.py --resume --sample-csv .nb1out/xai_aopc_sample.csv

# analyse, and verify the hand-rolled Wilcoxon against textbook cases
./venv/Scripts/python.exe scripts/xai_aopc_analyze.py --selftest
```

**Three things to remember.** AOPC on its own is meaningless — it needs a baseline. The
baseline must be **structure-matched** (`random7x7`), because `randompixel` wins by
breaking the image rather than by explaining it. And a partial run of a class-ordered
sample is **not a result**.
