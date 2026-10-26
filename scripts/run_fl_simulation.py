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
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.federated.server import weighted_average


class FedAvgWithCheckpoint(FedAvg):
    """FedAvg that captures the latest aggregated parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated and aggregated[0] is not None:
            self.latest_parameters = aggregated[0]
        return aggregated


from src.training.train_utils import (
    build_loss_weights,
    build_optimizer,
    create_model,
    evaluate,
    get_train_transform,
    get_val_transform,
    make_dataloader,
    train_one_epoch,
)


_SIM_CONFIG: dict = {}


def _make_client_data(data_dir: Path, shard_csv: Path) -> tuple[str, str]:
    """Copy shard images into a temp train/val split and return the paths."""
    with open(shard_csv, newline="") as f:
        shard_paths = {row["path"] for row in csv.DictReader(f)}

    work_dir = Path(tempfile.mkdtemp(prefix=f"fl_{shard_csv.stem}_"))
    train_dir = work_dir / "train"
    val_dir = work_dir / "val"

    for path_str in shard_paths:
        src = data_dir / path_str
        if not src.exists():
            continue
        parts = Path(path_str).parts
        original_split = parts[0]
        label = parts[1]
        dest_root = train_dir if original_split == "train" else val_dir
        dest_dir = dest_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(str(src), str(dest))

    return str(train_dir), str(val_dir)


def client_fn(cid: str) -> fl.client.NumPyClient:
    """Factory for flower's start_simulation — creates one hospital client."""
    from src.federated.client import SkinSureClient

    cfg = _SIM_CONFIG
    partition_id = int(cid)

    hospital_names = ["hospital_a", "hospital_b", "hospital_c"]
    hospital_name = hospital_names[partition_id]
    shard_csv = Path(cfg["shard_dir"]) / f"{hospital_name}.csv"

    train_dir, val_dir = _make_client_data(Path(cfg["data_dir"]), shard_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg["backbone"], num_classes=5, pretrained=False)
    model = model.to(device)

    _, classes = make_dataloader(
        train_dir, get_val_transform(224), batch_size=1, is_train=False,
    )

    return SkinSureClient(
        model=model,
        train_dir=train_dir,
        val_dir=val_dir,
        device=device,
        classes=classes,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
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

    global _SIM_CONFIG
    _SIM_CONFIG = {
        "backbone": args.backbone,
        "data_dir": str(args.data_dir),
        "shard_dir": str(args.shard_dir),
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    initial_model = create_model(args.backbone, num_classes=5, pretrained=True)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )

    strategy = FedAvgWithCheckpoint(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if torch.cuda.is_available():
        client_resources["num_gpus"] = 0.33

    print(f"Starting FL simulation: {args.backbone}, {args.num_rounds} rounds, "
          f"{args.local_epochs} local epochs, 3 hospitals")
    print(f"Shard dir: {args.shard_dir}")
    print(f"Client resources: {client_resources}\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    print("\n--- FL training complete ---")
    if history.metrics_distributed:
        for key, vals in history.metrics_distributed.items():
            if vals:
                last_round, last_val = vals[-1]
                print(f"  Round {last_round} {key}: {last_val:.4f}")

    print("\n--- Evaluating final global model on global test set ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.backbone, num_classes=5, pretrained=False)

    final_ndarrays = None
    if strategy.latest_parameters is not None:
        final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)

    if final_ndarrays is not None:
        state_dict = OrderedDict(
            (k, torch.tensor(np.array(v), dtype=param.dtype))
            for (k, param), v in zip(model.state_dict().items(), final_ndarrays)
        )
        model.load_state_dict(state_dict, strict=True)

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
