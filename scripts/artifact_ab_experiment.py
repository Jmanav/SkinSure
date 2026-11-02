"""Paired A/B experiment: does artifact removal help or hurt the ensemble?

Scores every image in the de-duplicated clean 565-image test set twice through the
same v2 ensemble: (A) baseline (resize + ImageNet norm only, exactly what the models
were trained on) and (B) `remove_artifacts` applied first. Writes raw per-image rows
to `.nb1out/artifact_ab_raw.csv` so the analysis never needs the models reloaded.

Also records mask diagnostics for EVERY image regardless of whether the step fired:
the combined mask fraction plus the per-detector (ink / glare / ruler) fractions. That
is what turns "it moved accuracy by X" into a root-cause finding.

The image list is read from `.nb1out/hair_ab_raw.csv` (one row per clean-set image)
so these numbers are directly comparable with the hair-removal A/B.

Does NOT read or modify `configs/preprocessing.yaml` -- the step is driven directly.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ensemble import SkinEnsemble  # noqa: E402
from src.preprocessing.artifact_removal import (  # noqa: E402
    MAX_MASK_FRACTION,
    _glare_mask,
    _ink_mask,
    _ruler_mask,
    remove_artifacts,
)
from src.preprocessing.inference_transform import preprocess_image  # noqa: E402
from src.utils.config import get_checkpoint_paths, get_model_config  # noqa: E402

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
HAIR_RAW = ROOT / ".nb1out" / "hair_ab_raw.csv"
OUT = ROOT / ".nb1out" / "artifact_ab_raw.csv"


def build_file_list() -> tuple[list[tuple[Path, str]], str]:
    """Image list taken verbatim from the hair A/B raw rows (the clean 565 set)."""
    if not HAIR_RAW.exists():
        sys.exit(f"missing {HAIR_RAW} -- needed for the exact 565-image list")
    rows: list[tuple[Path, str]] = []
    missing = 0
    with HAIR_RAW.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            p = Path(rec["path"])
            if not p.is_file():
                missing += 1
                continue
            rows.append((p, rec["true_class"]))
    basis = (f"clean de-duplicated 565 set, image list reused from {HAIR_RAW.name} "
             f"(n={len(rows)}, missing_on_disk={missing})")
    return rows, basis


@torch.no_grad()
def probs_of(ens: SkinEnsemble, img: Image.Image) -> np.ndarray:
    x = preprocess_image(img).to(ens.device, memory_format=torch.channels_last)
    w = ens.weights
    p = (
        w[0] * F.softmax(ens.backbones["efficientnet"](x), dim=1)
        + w[1] * F.softmax(ens.backbones["convnext"](x), dim=1)
        + w[2] * F.softmax(ens.backbones["swin"](x), dim=1)
    )
    return p.squeeze(0).cpu().numpy().astype(np.float64)


def mask_diagnostics(img: Image.Image) -> dict[str, float]:
    """Per-detector and combined mask coverage as a fraction of the frame.

    Recomputed here (same functions the step calls) so the diagnostics exist for
    skipped images too, where the step's own detail string only reports the total.
    """
    rgb = np.asarray(img.convert("RGB"))
    ink = _ink_mask(rgb) > 0
    glare = _glare_mask(rgb) > 0
    ruler = _ruler_mask(rgb) > 0
    comb = ink | glare | ruler
    n = float(rgb.shape[0] * rgb.shape[1])
    return {
        "frac_combined": float(comb.sum()) / n,
        "frac_ink": float(ink.sum()) / n,
        "frac_glare": float(glare.sum()) / n,
        "frac_ruler": float(ruler.sum()) / n,
        "px_h": rgb.shape[0],
        "px_w": rgb.shape[1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    files, basis = build_file_list()
    if args.limit:
        files = files[: args.limit]
        basis += f" -- LIMITED to first {args.limit}"
    print(f"basis: {basis}", flush=True)
    print(f"MAX_MASK_FRACTION = {MAX_MASK_FRACTION}", flush=True)

    w = get_model_config().weights
    ens = SkinEnsemble(get_checkpoint_paths(),
                       weights=(w["efficientnet"], w["convnext"], w["swin"]))
    for m in ens.backbones.values():
        m.eval()
    print(f"ensemble on {ens.device}, classes={ens.classes}", flush=True)
    assert ens.classes == CLASSES, ens.classes

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (["path", "true_class", "pred_a", "pred_b", "artifact_applied", "artifact_note",
               "frac_combined", "frac_ink", "frac_glare", "frac_ruler", "px_h", "px_w"]
              + [f"a_{c}" for c in CLASSES] + [f"b_{c}" for c in CLASSES])

    t0 = time.time()
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for i, (path, true_cls) in enumerate(files, 1):
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")

            pa = probs_of(ens, img)
            diag = mask_diagnostics(img)
            step = remove_artifacts(img)
            pb = probs_of(ens, step.image)

            wr.writerow(
                [str(path), true_cls, CLASSES[int(pa.argmax())], CLASSES[int(pb.argmax())],
                 int(bool(step.applied)), step.detail,
                 f"{diag['frac_combined']:.8f}", f"{diag['frac_ink']:.8f}",
                 f"{diag['frac_glare']:.8f}", f"{diag['frac_ruler']:.8f}",
                 diag["px_h"], diag["px_w"]]
                + [f"{v:.8f}" for v in pa] + [f"{v:.8f}" for v in pb]
            )
            if i % 20 == 0 or i == len(files):
                el = time.time() - t0
                fh.flush()
                print(f"{i}/{len(files)}  {el:.0f}s  eta {el / i * (len(files) - i):.0f}s",
                      flush=True)

    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
    print(f"BASIS_LINE: {basis}", flush=True)


if __name__ == "__main__":
    main()
