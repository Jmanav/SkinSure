"""Train a centralized baseline on the trainable pool to establish the target for FL.

Usage:
    python scripts/train_baseline.py --data-dir <path-to-presplit-dataset> --backbone efficientnet

Uses the trainable pool (data/splits/trainable_pool.csv) with the original
train/val split preserved from the presplit dataset. Evaluates on the global
test set (data/splits/global_test.csv).

The centralized baseline's metrics are the numbers federated training must approach.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from src.training.train_utils import (
    DEFAULT_CFG,
    create_model,
    evaluate,
    build_loss_weights,
    get_train_transform,
    get_val_transform,
    make_dataloader,
    run_training,
)


def load_manifest(csv_path: Path) -> set[str]:
    with open(csv_path, newline="") as f:
        return {row["path"] for row in csv.DictReader(f)}


def build_split_dirs(
    data_dir: Path,
    pool_paths: set[str],
    test_paths: set[str],
    work_dir: Path,
) -> tuple[Path, Path, Path]:
    """Symlink images into train/val/test ImageFolder trees under work_dir.

    Uses the original presplit structure: images in train/ go to train,
    images in val/ go to val. Global test images go to test/.
    """
    train_dir = work_dir / "train"
    val_dir = work_dir / "val"
    test_dir = work_dir / "test"

    for path_str in pool_paths:
        src = data_dir / path_str
        if not src.exists():
            continue
        parts = Path(path_str).parts
        split = parts[0]
        label = parts[1]

        if split == "train":
            dest_dir = train_dir / label
        else:
            dest_dir = val_dir / label

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(str(src), str(dest))

    for path_str in test_paths:
        src = data_dir / path_str
        if not src.exists():
            continue
        parts = Path(path_str).parts
        label = parts[1]
        dest_dir = test_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(str(src), str(dest))

    return train_dir, val_dir, test_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized baseline.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root of presplit dataset (with train/val/test)")
    parser.add_argument("--backbone", default="efficientnet",
                        choices=["efficientnet", "convnext", "swin"])
    parser.add_argument("--pool", type=Path, default=Path("data/splits/trainable_pool.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/splits/global_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/baselines"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pool_paths = load_manifest(args.pool)
    test_paths = load_manifest(args.test_set)
    print(f"Trainable pool: {len(pool_paths)} | Global test: {len(test_paths)}")

    work_dir = Path(tempfile.mkdtemp(prefix="skinsure_baseline_"))
    print(f"Building ImageFolder trees in {work_dir}...")

    train_dir, val_dir, test_dir = build_split_dirs(
        args.data_dir, pool_paths, test_paths, work_dir
    )

    cfg = dict(DEFAULT_CFG)
    if args.epochs:
        cfg["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    img_size = cfg["img_size"]
    train_loader, classes = make_dataloader(
        str(train_dir), get_train_transform(img_size),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], is_train=True,
    )
    val_loader, _ = make_dataloader(
        str(val_dir), get_val_transform(img_size),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], is_train=False,
    )
    test_loader, _ = make_dataloader(
        str(test_dir), get_val_transform(img_size),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], is_train=False,
    )

    print(f"\nClasses: {classes}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = create_model(args.backbone, num_classes=len(classes), pretrained=True)
    model = model.to(device)

    print(f"\n--- Training {args.backbone} baseline ---")
    result = run_training(model, train_loader, val_loader, classes, device, cfg)

    if result["best_state_dict"]:
        model.load_state_dict(result["best_state_dict"])
        model = model.to(device)

    import torch.nn as nn
    from collections import Counter
    class_counts = Counter(test_loader.dataset.targets)
    weights_tensor = build_loss_weights(class_counts, classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    print(f"\n--- Evaluating on global test set ({len(test_loader.dataset)} images) ---")
    test_metrics = evaluate(model, test_loader, criterion, device, classes, use_amp=False)

    print(f"\nCentralized Baseline Results ({args.backbone}):")
    print(f"  Best composite (val):   {result['best_composite']:.4f}")
    print(f"  Best val_acc:           {result['best_val_acc']*100:.2f}%")
    print(f"  Best melanoma_sens:     {result['best_melanoma_sens']*100:.2f}%")
    print(f"\nGlobal Test Set:")
    print(f"  Test acc:               {test_metrics['val_acc']*100:.2f}%")
    print(f"  Melanoma sensitivity:   {test_metrics['melanoma_sensitivity']*100:.2f}%")
    print(f"  BCC sensitivity:        {test_metrics['bcc_sensitivity']*100:.2f}%")
    print(f"  Composite:              {test_metrics['composite']:.4f}")
    print(f"\n  Per-class sensitivity:")
    for cls, sens in sorted(test_metrics["per_class_sensitivity"].items()):
        print(f"    {cls:<25} {sens*100:.2f}%")

    args.output.mkdir(parents=True, exist_ok=True)
    save_path = args.output / f"{args.backbone}_baseline.pth"
    torch.save({
        "model_state": result["best_state_dict"],
        "classes": classes,
        "backbone": args.backbone,
        "val_metrics": {
            "best_composite": result["best_composite"],
            "best_val_acc": result["best_val_acc"],
            "best_melanoma_sens": result["best_melanoma_sens"],
        },
        "test_metrics": test_metrics,
        "history": result["history"],
        "config": cfg,
    }, save_path)
    print(f"\nSaved to {save_path}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
