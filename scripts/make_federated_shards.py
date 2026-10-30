"""Split the trainable pool into N=3 hospital shards for federated learning simulation.

Usage:
    python scripts/make_federated_shards.py [--strategy iid|noniid] [--pool data/splits/trainable_pool.csv]

Outputs shard manifests to data/splits/shards/:
    hospital_a.csv, hospital_b.csv, hospital_c.csv

Each CSV has columns: path, label

Strategies:
    iid    - random stratified split, same class balance in each shard
    noniid - skewed by hospital specialization:
             Hospital A: heavy Melanoma + BCC (urban dermatology center)
             Hospital B: heavy Tinea + Eczema (tropical clinic)
             Hospital C: mixed (multi-specialty hospital)

See fedler.md Phase 1 for rationale.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
MIN_PER_CLASS = 200

HOSPITAL_PROFILES = {
    "hospital_a": {
        "description": "Large urban dermatology center - heavy Melanoma + BCC",
        "weights": {
            "Melanoma": 0.50,
            "Basal_Cell_Carcinoma": 0.45,
            "Keratosis": 0.30,
            "Eczema": 0.30,
            "Tinea": 0.30,
        },
    },
    "hospital_b": {
        "description": "Tropical area clinic - heavy Tinea + Eczema",
        "weights": {
            "Melanoma": 0.25,
            "Basal_Cell_Carcinoma": 0.25,
            "Keratosis": 0.30,
            "Eczema": 0.45,
            "Tinea": 0.50,
        },
    },
    "hospital_c": {
        "description": "Multi-specialty hospital - mixed distribution",
        "weights": {
            "Melanoma": 0.25,
            "Basal_Cell_Carcinoma": 0.30,
            "Keratosis": 0.40,
            "Eczema": 0.25,
            "Tinea": 0.20,
        },
    },
}


def load_pool(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [(row["path"], row["label"]) for row in reader]


def iid_split(
    samples: list[tuple[str, str]], n_shards: int, rng: np.random.Generator
) -> list[list[tuple[str, str]]]:
    """Random stratified split into n_shards with equal class balance."""
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    shards: list[list[tuple[str, str]]] = [[] for _ in range(n_shards)]
    for label in sorted(by_class):
        items = by_class[label].copy()
        rng.shuffle(items)
        splits = np.array_split(items, n_shards)
        for i, chunk in enumerate(splits):
            shards[i].extend(chunk.tolist())

    return shards


def noniid_split(
    samples: list[tuple[str, str]], rng: np.random.Generator
) -> dict[str, list[tuple[str, str]]]:
    """Non-IID split following hospital profiles with minimum floor enforcement."""
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    hospital_names = list(HOSPITAL_PROFILES.keys())
    shards: dict[str, list[tuple[str, str]]] = {h: [] for h in hospital_names}

    for label in sorted(by_class):
        items = by_class[label].copy()
        rng.shuffle(items)
        n_total = len(items)

        raw_weights = [HOSPITAL_PROFILES[h]["weights"][label] for h in hospital_names]
        weight_sum = sum(raw_weights)
        fractions = [w / weight_sum for w in raw_weights]

        allocations = [max(MIN_PER_CLASS, int(n_total * f)) for f in fractions]

        while sum(allocations) > n_total:
            max_idx = int(np.argmax(allocations))
            allocations[max_idx] -= 1

        while sum(allocations) < n_total:
            min_idx = int(np.argmin(allocations))
            allocations[min_idx] += 1

        offset = 0
        for i, h in enumerate(hospital_names):
            shards[h].extend(items[offset:offset + allocations[i]])
            offset += allocations[i]

    return shards


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create federated hospital shards.")
    parser.add_argument("--strategy", choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--pool", type=Path, default=Path("data/splits/trainable_pool.csv"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/splits/shards/<strategy>)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.pool.exists():
        print(f"Error: {args.pool} not found. Run scripts/make_splits.py first.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or Path(f"data/splits/shards/{args.strategy}")
    rng = np.random.default_rng(args.seed)

    samples = load_pool(args.pool)
    print(f"Loaded {len(samples)} samples from {args.pool}")
    print(f"Strategy: {args.strategy}\n")

    if args.strategy == "iid":
        shard_list = iid_split(samples, 3, rng)
        names = ["hospital_a", "hospital_b", "hospital_c"]
        shard_dict = dict(zip(names, shard_list))
    else:
        shard_dict = noniid_split(samples, rng)

    all_classes = sorted(set(label for _, label in samples))

    for name, shard in shard_dict.items():
        out_path = output_dir / f"{name}.csv"
        write_csv(out_path, shard)

        desc = HOSPITAL_PROFILES.get(name, {}).get("description", "")
        print(f"{name} ({desc})")
        print(f"  Total: {len(shard)}")
        for cls in all_classes:
            n = sum(1 for _, l in shard if l == cls)
            flag = " (!) BELOW MIN" if n < MIN_PER_CLASS else ""
            print(f"    {cls:<25} {n:>5}{flag}")
        print(f"  -> {out_path}\n")


if __name__ == "__main__":
    main()
