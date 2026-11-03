"""Faithfulness evaluation: does Grad-CAM beat a random baseline on the deletion/AOPC test?

This is the first faithfulness measurement of the explanation layer (CLAUDE.md gap 10).
The repo's choice of Grad-CAM on /predict rests on latency and output shape, never on
measured faithfulness -- this script measures it.

Procedure (ONE procedure for every method, so the scores are comparable):
  * Sample is the de-duplicated 565-image clean test set, stratified N per class,
    drawn with a fixed seed. Membership comes from `.nb1out/hair_ab_raw.csv`, which
    holds exactly one row per clean-set image.
  * Each image goes through `preprocess_image` only (resize + ImageNet norm), i.e.
    exactly what the models were trained on. No pixel-changing preprocessing steps.
  * The class explained is the ensemble's PREDICTED class (argmax), not the true
    label: faithfulness asks "did the map find the pixels this prediction used".
  * `deletion_curve` blanks the top-k pixels by |heatmap| at 10%..100% and records
    p(predicted class). Blanking writes 0.0 in NORMALISED space = the ImageNet mean
    colour. AOPC = mean drop from the undeleted confidence. Higher = more faithful.

Heatmaps scored per image:
  gradcam      weighted ensemble Grad-CAM (7x7 per backbone, bilinear to 224)
  random7x7    uniform noise on a 7x7 grid, upsampled to 224 with the SAME bilinear
               interpolate call Grad-CAM uses. This is the FAIR baseline: it matches
               Grad-CAM's spatial granularity, so deletion removes contiguous blobs
               of comparable size in both cases.
  randompixel  uniform noise at full 224x224 resolution. Reported for contrast only;
               it is a weaker straw man, because scattering single deleted pixels
               across the frame is something a CNN tolerates far better than losing
               one contiguous region. The 7x7 comparison is the honest headline.

Several random seeds are scored per image (--random-seeds) so the baseline is not a
single draw. Writes one row per (image, method, seed) to `.nb1out/xai_aopc_raw.csv`.
No API spend: this is all local torch.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explainability.evaluation import (  # noqa: E402
    DELETION_PERCENTS,
    aopc,
    deletion_curve,
    ensemble_probs_fn,
)
from src.explainability.gradcam import ensemble_heatmap  # noqa: E402
from src.models.ensemble import SkinEnsemble  # noqa: E402
from src.preprocessing.inference_transform import preprocess_image  # noqa: E402
from src.utils.config import get_checkpoint_paths, get_model_config  # noqa: E402

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
CLEAN_SET = ROOT / ".nb1out" / "hair_ab_raw.csv"
OUT = ROOT / ".nb1out" / "xai_aopc_raw.csv"
SAMPLE = ROOT / ".nb1out" / "xai_aopc_sample.csv"

# Grad-CAM's native resolution in all three backbones at 224px input.
CAM_GRID = 7


def batched_deletion_curve(ensemble, image_tensor, heatmap, class_idx,
                           percents=DELETION_PERCENTS) -> list[float]:
    """Same curve as `src.explainability.evaluation.deletion_curve`, one batched forward.

    `deletion_curve` runs one ensemble forward per deletion level (11 per heatmap). The
    levels are independent, so stacking them into a single batch gives identical numbers
    for a fraction of the wall time -- which matters on this 2-core dev CPU. Blanking is
    still 0.0 in normalised space (the ImageNet mean colour), the ordering is still
    descending |heatmap| with the same stable argsort, and the per-backbone weighted
    softmax average is the same one `ensemble_probs_fn` uses.

    `test_batched_matches_reference()` asserts agreement with the reference to 1e-5.
    """
    height, width = heatmap.shape
    order = np.argsort(-np.abs(heatmap).ravel(), kind="stable")
    variants = []
    for pct in percents:
        k = int(order.size * pct / 100)
        modified = image_tensor.clone()
        if k:
            rows, cols = np.divmod(order[:k], width)
            modified[:, :, rows, cols] = 0.0
        variants.append(modified)
    batch = torch.cat(variants, dim=0).to(ensemble.device)
    with torch.no_grad():
        total = 0.0
        for model, weight in zip(ensemble.backbones.values(), ensemble.weights):
            total = total + weight * torch.softmax(model(batch), dim=1)
    return [float(v) for v in total[:, class_idx].cpu().numpy()]


def load_clean_set() -> dict[str, list[Path]]:
    """One row per de-duplicated clean-set image; group the 565 paths by true class."""
    by_class: dict[str, list[Path]] = {c: [] for c in CLASSES}
    with CLEAN_SET.open(encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            cls = rec["true_class"]
            if cls in by_class:
                by_class[cls].append(Path(rec["path"]))
    return by_class


def sample_stratified(per_class: int, seed: int) -> list[tuple[Path, str]]:
    """Deterministic stratified sample: `per_class` images from each of the 5 classes."""
    by_class = load_clean_set()
    rows: list[tuple[Path, str]] = []
    for cls in CLASSES:
        pool = sorted(by_class[cls], key=lambda p: p.name)  # stable order before sampling
        rng = random.Random(f"{seed}:{cls}")
        take = min(per_class, len(pool))
        if take < per_class:
            print(f"  WARNING: {cls} has only {len(pool)} clean images, wanted {per_class}")
        rows.extend((p, cls) for p in rng.sample(pool, take))
    return rows


def load_sample_csv(path: Path) -> list[tuple[Path, str]]:
    """Read a recorded sample manifest (`true_class,filename,path`) back in file order.

    Pins the run to the exact image list that was recorded, rather than re-deriving it
    from the seed. Same images either way while the sampling code is unchanged, but this
    cannot drift if anyone later touches `sample_stratified`.
    """
    rows: list[tuple[Path, str]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            rows.append((Path(rec["path"]), rec["true_class"]))
    return rows


def already_scored(out: Path) -> set[str]:
    """Image paths that already have rows in `out`, so a resumed run can skip them.

    Only images whose LAST row is present are trusted: a run interrupted midway through
    an image would leave a partial set of methods, so the final image in the file is
    re-scored rather than assumed complete.
    """
    if not out.exists() or out.stat().st_size == 0:
        return set()
    seen: list[str] = []
    with out.open(encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            if not seen or seen[-1] != rec["path"]:
                seen.append(rec["path"])
    if not seen:
        return set()
    return set(seen[:-1])   # drop the last (possibly incomplete) image


def random_map(rng: np.random.Generator, grid: int) -> np.ndarray:
    """Uniform noise on a `grid`x`grid` map, bilinearly upsampled to 224x224.

    Uses the same `F.interpolate(..., mode="bilinear", align_corners=False)` call as
    `grad_cam`, so the baseline's spatial granularity matches Grad-CAM's exactly.
    """
    if grid >= 224:
        return rng.random((224, 224))
    small = torch.from_numpy(rng.random((1, 1, grid, grid)).astype(np.float32))
    up = F.interpolate(small, size=(224, 224), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=20261102)
    parser.add_argument("--random-seeds", type=int, default=3,
                        help="how many independent random-baseline draws per image")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, default=2,
                        help="torch CPU threads; keep low when other jobs share the box")
    parser.add_argument("--reference-scorer", action="store_true",
                        help="use the unbatched evaluation.deletion_curve (slower, same result)")
    parser.add_argument("--sample-csv", type=Path, default=SAMPLE,
                        help="recorded sample manifest to reuse; re-derives from the seed "
                             "if the file is absent")
    parser.add_argument("--resume", action="store_true",
                        help="skip images already present in --out and append to it")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    if args.sample_csv and args.sample_csv.exists():
        files = load_sample_csv(args.sample_csv)
        # The manifest is the authority, but confirm it still matches the seed so a stale
        # file cannot silently change which images the run covers.
        derived = [p.name for p, _ in sample_stratified(args.per_class, args.sample_seed)]
        match = sorted(derived) == sorted(p.name for p, _ in files)
        print(f"Reusing sample manifest {args.sample_csv} ({len(files)} images); "
              f"seed {args.sample_seed} reproduces it: {match}")
        if not match:
            print("  WARNING: manifest does not match the seed -- using the MANIFEST "
                  "(recorded sample wins), but the two have diverged.")
    else:
        files = sample_stratified(args.per_class, args.sample_seed)
        print(f"Sampled {len(files)} images from the de-duplicated 565-image clean test set "
              f"(seed {args.sample_seed}, {args.per_class}/class)")

    done: set[str] = set()
    if args.resume:
        done = already_scored(args.out)
        if done:
            # Drop the trailing partial image's rows so the re-scored image is not
            # counted twice, then append from there.
            kept = [r for r in args.out.read_text(encoding="utf-8").splitlines()
                    if r.split(",")[0] in done or r.startswith("path,")]
            args.out.write_text("\n".join(kept) + "\n", encoding="utf-8")
            files = [(p, c) for p, c in files if str(p) not in done]
            print(f"Resuming: {len(done)} image(s) already scored, {len(files)} remaining")
    if not files:
        print("Nothing left to score.")
        return

    w = get_model_config().weights
    ensemble = SkinEnsemble(get_checkpoint_paths(),
                            weights=(w["efficientnet"], w["convnext"], w["swin"]))
    probs_fn = ensemble_probs_fn(ensemble)

    def score(tensor, hm, class_idx) -> list[float]:
        if args.reference_scorer:
            return deletion_curve(probs_fn, tensor, hm, class_idx)
        return batched_deletion_curve(ensemble, tensor, hm, class_idx)

    verified = args.reference_scorer   # first image cross-checks batched vs reference

    args.out.parent.mkdir(parents=True, exist_ok=True)
    appending = bool(done)
    fh = args.out.open("a" if appending else "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if not appending:
        writer.writerow(
            ["path", "true_class", "pred_class", "correct", "pred_conf", "method", "seed",
             "aopc", "secs"]
            + [f"p{p}" for p in DELETION_PERCENTS]
        )

    t_start = time.time()
    for i, (path, true_cls) in enumerate(files, 1):
        tensor = preprocess_image(Image.open(path).convert("RGB"))
        base_probs = probs_fn(tensor)
        class_idx = int(np.argmax(base_probs))
        pred_cls = ensemble.classes[class_idx]
        pred_conf = float(base_probs[class_idx])

        # Grad-CAM, scored once (deterministic).
        t0 = time.time()
        cam = ensemble_heatmap(ensemble, tensor, class_idx)
        cam_secs = time.time() - t0
        jobs: list[tuple[str, int, np.ndarray, float]] = [("gradcam", -1, cam, cam_secs)]

        # Random baselines, several independent draws each. The seed is derived from the
        # file name so a re-run reproduces the same maps for the same image.
        for s in range(args.random_seeds):
            for name, grid in (("random7x7", CAM_GRID), ("randompixel", 224)):
                rng = np.random.default_rng(
                    abs(hash((path.name, name, s))) % (2**32)
                )
                jobs.append((name, s, random_map(rng, grid), 0.0))

        if not verified:
            # Cross-check the batched scorer against the repo's reference implementation
            # once, on the first real image and the real Grad-CAM map, before trusting it.
            ref = deletion_curve(probs_fn, tensor, cam, class_idx)
            got = batched_deletion_curve(ensemble, tensor, cam, class_idx)
            delta = max(abs(a - b) for a, b in zip(ref, got))
            assert delta < 1e-5, f"batched scorer disagrees with reference by {delta:.2e}"
            print(f"  batched scorer verified against evaluation.deletion_curve "
                  f"(max abs delta {delta:.2e})")
            verified = True

        line = []
        for name, seed, hm, build_secs in jobs:
            t0 = time.time()
            curve = score(tensor, hm, class_idx)
            aopc_score = aopc(curve)
            writer.writerow(
                [str(path), true_cls, pred_cls, int(pred_cls == true_cls),
                 f"{pred_conf:.6f}", name, seed, f"{aopc_score:.6f}",
                 f"{build_secs + time.time() - t0:.3f}"]
                + [f"{c:.6f}" for c in curve]
            )
            if seed <= 0:
                line.append(f"{name}={aopc_score:.3f}")
        fh.flush()
        elapsed = time.time() - t_start
        print(f"[{i}/{len(files)}] {true_cls}/{path.name} pred={pred_cls} "
              f"({pred_conf:.3f}) " + "  ".join(line)
              + f"  cam={cam_secs:.1f}s  elapsed={elapsed/60:.1f}m")

    fh.close()
    print(f"\nWrote {args.out}  ({(time.time() - t_start)/60:.1f} min total, "
          f"{(time.time() - t_start)/len(files):.1f} s/image)")


if __name__ == "__main__":
    main()
