"""Run a federated learning simulation with 3 hospital clients on one machine.

Usage:
    python scripts/run_fl_simulation.py \
        --data-dir data/raw/IPD_Datasets_Final_Split \
        --shard-dir data/splits/shards/noniid \
        --backbone efficientnet \
        --num-rounds 5 \
        --local-epochs 2

Evaluates the final global model on data/splits/global_test.csv.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

import flwr as fl
from flwr.simulation import run_simulation

from src.federated.client import app as client_app
from src.federated.server import app as server_app
from src.training.train_utils import (
    build_loss_weights,
    create_model,
    evaluate,
    get_val_transform,
    make_dataloader,
)


def evaluate_global_model(
    model: nn.Module,
    data_dir: Path,
    test_csv: Path,
    device: torch.device,
) -> dict:
    """Evaluate the global model on the held-out global test set."""
    with open(test_csv, newline="") as f:
        test_paths = {row["path"] for row in csv.DictReader(f)}

    work_dir = Path(tempfile.mkdtemp(prefix="fl_eval_"))
    test_dir = work_dir / "test"

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

    loader, classes = make_dataloader(
        str(test_dir), get_val_transform(224),
        batch_size=16, is_train=False,
    )

    class_counts = Counter(loader.dataset.targets)
    weights = build_loss_weights(class_counts, classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    metrics = evaluate(model, loader, criterion, device, classes, use_amp=False)
    shutil.rmtree(work_dir, ignore_errors=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FL simulation.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("data/splits/shards/noniid"))
    parser.add_argument("--backbone", default="efficientnet",
                        choices=["efficientnet", "convnext", "swin"])
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--test-set", type=Path, default=Path("data/splits/global_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/federated"))
    args = parser.parse_args()

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if torch.cuda.is_available():
        backend_config["client_resources"]["num_gpus"] = 0.33

    run_config = {
        "backbone": args.backbone,
        "data-dir": str(args.data_dir),
        "shard-dir": str(args.shard_dir),
        "local-epochs": str(args.local_epochs),
        "batch-size": str(args.batch_size),
        "lr": str(args.lr),
        "num-rounds": str(args.num_rounds),
        "fraction-fit": "1.0",
        "fraction-evaluate": "1.0",
        "min-fit-clients": "3",
        "min-evaluate-clients": "3",
        "min-available-clients": "3",
    }

    print(f"Starting FL simulation: {args.backbone}, {args.num_rounds} rounds, "
          f"{args.local_epochs} local epochs, 3 hospitals")
    print(f"Shard dir: {args.shard_dir}")
    print(f"Backend: {backend_config}\n")

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_config=backend_config,
        run_config=run_config,
    )

    print("\n--- Evaluating final global model on global test set ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.backbone, num_classes=5, pretrained=False)
    model = model.to(device)

    test_metrics = evaluate_global_model(model, args.data_dir, args.test_set, device)

    print(f"\nFederated Results ({args.backbone}, {args.num_rounds} rounds):")
    print(f"  Test acc:             {test_metrics['val_acc']*100:.2f}%")
    print(f"  Melanoma sensitivity: {test_metrics['melanoma_sensitivity']*100:.2f}%")
    print(f"  BCC sensitivity:      {test_metrics['bcc_sensitivity']*100:.2f}%")
    print(f"  Composite:            {test_metrics['composite']:.4f}")
    print(f"\n  Per-class sensitivity:")
    for cls, sens in sorted(test_metrics["per_class_sensitivity"].items()):
        print(f"    {cls:<25} {sens*100:.2f}%")

    args.output.mkdir(parents=True, exist_ok=True)
    save_path = args.output / f"{args.backbone}_federated.pth"
    torch.save({
        "model_state": model.state_dict(),
        "backbone": args.backbone,
        "test_metrics": test_metrics,
        "fl_config": {
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "strategy": "FedAvg",
            "num_clients": 3,
            "shard_strategy": args.shard_dir.parent.name if args.shard_dir.parent.name in ("iid", "noniid") else str(args.shard_dir),
        },
    }, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
