"""Paired A/B experiment: does hair removal help or hurt the ensemble?

Scores every image in the de-duplicated clean test set twice through the same
ensemble: (A) baseline (resize + ImageNet norm only, exactly what the models were
trained on) and (B) `remove_hair` applied first. Writes raw per-image results to
`.nb1out/hair_ab_raw.csv` so the analysis never needs the models reloaded.

Does NOT read or modify `configs/preprocessing.yaml` -- the step is driven directly.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import PurePosixPath
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ensemble import SkinEnsemble  # noqa: E402
from src.preprocessing.hair_removal import remove_hair  # noqa: E402
from src.preprocessing.inference_transform import preprocess_image  # noqa: E402
from src.utils.config import get_checkpoint_paths, get_model_config  # noqa: E402

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
MANIFEST = Path(r"C:\Users\Cipher\Downloads\test_clean_manifest.csv")
TEST_ROOT = ROOT / "data" / "raw" / "IPD_Datasets_Final_Split" / "test"
OUT = ROOT / ".nb1out" / "hair_ab_raw.csv"


def build_file_list(per_class: int | None) -> tuple[list[tuple[Path, str]], str]:
    """Return [(path, true_class)] and a string describing the basis used."""
    on_disk: dict[str, dict[str, Path]] = {}
    for cls in CLASSES:
        on_disk[cls] = {p.name: p for p in (TEST_ROOT / cls).iterdir() if p.is_file()}

    rows: list[tuple[Path, str]] = []
    basis: str
    if MANIFEST.exists():
        missing = 0
        with MANIFEST.open(encoding="utf-8-sig") as fh:
            for rec in csv.DictReader(fh):
                # match the way .nb1out/_agree.py does it: parent dir name + filename
                man = PurePosixPath(rec["path"])
                cls, name = man.parent.name, man.name
                if cls not in on_disk:
                    continue
                hit = on_disk[cls].get(name)
                if hit is None:
                    missing += 1
                    continue
                rows.append((hit, cls))
        basis = f"clean de-duplicated manifest (n={len(rows)}, unmatched={missing})"
    else:
        for cls in CLASSES:
            rows.extend((p, cls) for p in sorted(on_disk[cls].values()))
        basis = f"CONTAMINATED full test dir (n={len(rows)}) -- manifest not found"

    if per_class is not None:
        rng = random.Random(42)
        sampled: list[tuple[Path, str]] = []
        for cls in CLASSES:
            pool = sorted([r for r in rows if r[1] == cls], key=lambda r: r[0].name)
            rng.shuffle(pool)
            sampled.extend(pool[:per_class])
        rows = sampled
        basis += f" -- stratified sample, seed 42, <={per_class}/class, n={len(rows)}"

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=None,
                    help="stratified sample size per class (seed 42); omit for the full set")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    files, basis = build_file_list(args.per_class)
    print(f"basis: {basis}", flush=True)

    w = get_model_config().weights
    ens = SkinEnsemble(get_checkpoint_paths(),
                       weights=(w["efficientnet"], w["convnext"], w["swin"]))
    for m in ens.backbones.values():
        m.eval()
    print(f"ensemble on {ens.device}, classes={ens.classes}", flush=True)
    assert ens.classes == CLASSES, ens.classes

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (["path", "true_class", "pred_a", "pred_b", "hair_applied", "hair_note"]
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
            step = remove_hair(img)
            pb = probs_of(ens, step.image)

            wr.writerow(
                [str(path), true_cls, CLASSES[int(pa.argmax())], CLASSES[int(pb.argmax())],
                 int(bool(step.applied)), step.detail]
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
