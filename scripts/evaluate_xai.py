"""Compare explanation faithfulness with the deletion/AOPC test.

Runs Grad-CAM (this repo) on every image under IMAGE_DIR/<class>/*.{jpg,jpeg,png} and, if
--heatmap-dir is given, also scores externally produced heatmaps (e.g. the SHAP .npy files from
the xai-shap notebooks) under the SAME procedure and baseline. A random heatmap is always
scored too: a method that does not beat it explains nothing.

    python scripts/evaluate_xai.py <image_dir> [--heatmap-dir DIR] [--limit N]
        [--heatmap-pattern "{cls}/{stem}_ensemble_shap.npy"]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.explainability.evaluation import aopc, deletion_curve, ensemble_probs_fn
from src.explainability.gradcam import ensemble_heatmap
from src.models.ensemble import SkinEnsemble
from src.preprocessing.inference_transform import preprocess_image
from src.utils.config import get_checkpoint_paths, get_model_config

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--heatmap-dir", type=Path, default=None)
    parser.add_argument("--heatmap-pattern", default="{cls}/{stem}_ensemble_shap.npy")
    parser.add_argument("--limit", type=int, default=None, help="max images per class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    w = get_model_config().weights
    ensemble = SkinEnsemble(
        get_checkpoint_paths(), weights=(w["efficientnet"], w["convnext"], w["swin"])
    )
    probs_fn = ensemble_probs_fn(ensemble)
    rng = np.random.default_rng(args.seed)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for class_dir in sorted(p for p in args.image_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTS)
        for path in files[: args.limit]:
            tensor = preprocess_image(Image.open(path).convert("RGB"))
            class_idx = int(np.argmax(probs_fn(tensor)))   # explain the predicted class
            maps = {
                "gradcam": ensemble_heatmap(ensemble, tensor, class_idx),
                "random": rng.random(tensor.shape[-2:]),
            }
            if args.heatmap_dir:
                ext = args.heatmap_dir / args.heatmap_pattern.format(cls=class_dir.name, stem=path.stem)
                if ext.exists():
                    maps["external"] = np.load(ext)
                else:
                    skipped += 1
            for name, hm in maps.items():
                scores[name][class_dir.name].append(aopc(deletion_curve(probs_fn, tensor, hm, class_idx)))
            print(f"  {class_dir.name}/{path.name}: " + "  ".join(
                f"{n}={scores[n][class_dir.name][-1]:.3f}" for n in maps))

    print("\nAOPC (higher = more faithful; compare against 'random')")
    for name, per_class in scores.items():
        allv = [v for vals in per_class.values() for v in vals]
        print(f"  {name:9s} overall {np.mean(allv):.4f}  (n={len(allv)})")
        for cls, vals in sorted(per_class.items()):
            print(f"      {cls:22s} {np.mean(vals):.4f}  (n={len(vals)})")
    if skipped:
        print(f"\n{skipped} image(s) had no matching external heatmap and were scored without it.")


if __name__ == "__main__":
    main()
