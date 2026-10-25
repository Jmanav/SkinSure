"""Create canonical train/val/test split manifests from the raw dataset.

Usage:
    python scripts/make_splits.py --data-dir <path-to-dataset-root>

The dataset root should contain class-named subdirectories, each holding images:
    <data-dir>/
        Basal_Cell_Carcinoma/
        Eczema/
        Keratosis/
        Melanoma/
        Tinea/

If the dataset is already split into train/val/test folders (the Colab layout),
pass --presplit and the script expects:
    <data-dir>/train/<class>/...
    <data-dir>/val/<class>/...
    <data-dir>/test/<class>/...

Outputs:
    data/splits/global_test.csv     — ~15% stratified holdout (never used for training)
    data/splits/trainable_pool.csv  — remaining ~85% (used for centralized or federated training)

Each CSV has columns: path, label
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

EXPECTED_CLASSES = {"Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
TEST_FRACTION = 0.15
SEED = 42


def collect_flat(data_dir: Path) -> list[tuple[str, str]]:
    """Collect (relative_path, label) from a flat class-folder layout."""
    samples = []
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        if label not in EXPECTED_CLASSES:
            print(f"  [skip] unexpected directory: {label}")
            continue
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS:
                samples.append((str(img.relative_to(data_dir)), label))
    return samples


def collect_presplit(data_dir: Path) -> list[tuple[str, str]]:
    """Collect from train/val/test subdirectories, flattening them."""
    samples = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  [warn] {split}/ not found, skipping")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name
            if label not in EXPECTED_CLASSES:
                print(f"  [skip] unexpected directory: {split}/{label}")
                continue
            for img in sorted(cls_dir.rglob("*")):
                if img.suffix.lower() in IMAGE_EXTS:
                    rel = str(img.relative_to(data_dir))
                    samples.append((rel, label))
    return samples


def stratified_split(
    samples: list[tuple[str, str]], test_frac: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratified split into (trainable_pool, global_test)."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    pool, test = [], []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        test.extend(items[:n_test])
        pool.extend(items[n_test:])
    return pool, test


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical data split manifests.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root of the image dataset")
    parser.add_argument("--presplit", action="store_true",
                        help="Dataset has train/val/test subdirectories")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"),
                        help="Where to write the CSVs (default: data/splits)")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION,
                        help=f"Fraction for global test set (default: {TEST_FRACTION})")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting images from {args.data_dir} ({'presplit' if args.presplit else 'flat'})...")
    if args.presplit:
        samples = collect_presplit(args.data_dir)
    else:
        samples = collect_flat(args.data_dir)

    if not samples:
        print("Error: no images found", file=sys.stderr)
        sys.exit(1)

    found_classes = sorted(set(label for _, label in samples))
    missing = EXPECTED_CLASSES - set(found_classes)
    if missing:
        print(f"Error: missing classes: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Total images: {len(samples)}")
    for cls in found_classes:
        n = sum(1 for _, l in samples if l == cls)
        print(f"  {cls:<25} {n:>5}")

    pool, test = stratified_split(samples, args.test_fraction, args.seed)

    pool_path = args.output_dir / "trainable_pool.csv"
    test_path = args.output_dir / "global_test.csv"
    write_csv(pool_path, pool)
    write_csv(test_path, test)

    print(f"\nTrainable pool: {len(pool):>5} → {pool_path}")
    print(f"Global test:    {len(test):>5} → {test_path}")

    print("\nPer-class breakdown:")
    print(f"  {'Class':<25} {'Pool':>6} {'Test':>6}")
    for cls in found_classes:
        np_ = sum(1 for _, l in pool if l == cls)
        nt = sum(1 for _, l in test if l == cls)
        print(f"  {cls:<25} {np_:>6} {nt:>6}")


if __name__ == "__main__":
    main()
