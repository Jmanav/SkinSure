"""Flower client representing one hospital in the federated learning simulation.

Each client wraps a single backbone (EfficientNet, ConvNeXt, or Swin) and trains
on its own hospital shard. Only model weight updates are communicated — never raw
images.

See fedler.md Phase 2 for design rationale.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.training.train_utils import (
    build_loss_weights,
    build_optimizer,
    build_scheduler,
    create_model,
    evaluate,
    get_train_transform,
    get_val_transform,
    make_dataloader,
    train_one_epoch,
    EXPECTED_CLASSES,
)


class SkinSureClient(NumPyClient):
    """Flower client for one hospital. Trains a single backbone on local data."""

    def __init__(
        self,
        model: nn.Module,
        train_dir: str,
        val_dir: str,
        device: torch.device,
        classes: list[str],
        local_epochs: int = 2,
        batch_size: int = 16,
        lr: float = 2e-4,
    ) -> None:
        self.model = model
        self.device = device
        self.classes = classes
        self.local_epochs = local_epochs
        self.lr = lr

        img_size = 224
        self.train_loader, _ = make_dataloader(
            train_dir, get_train_transform(img_size),
            batch_size=batch_size, is_train=True,
        )
        self.val_loader, _ = make_dataloader(
            val_dir, get_val_transform(img_size),
            batch_size=batch_size, is_train=False,
        )

        class_counts = Counter(self.train_loader.dataset.targets)
        weights_tensor = build_loss_weights(class_counts, classes, device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = OrderedDict(
            (k, torch.tensor(v, dtype=param.dtype))
            for (k, param), v in zip(self.model.state_dict().items(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)
        self.model.to(self.device)

        optimizer = build_optimizer(
            self.model, self.lr, weight_decay=2e-4, lr_decay_rate=0.1
        )

        for epoch in range(self.local_epochs):
            loss, acc = train_one_epoch(
                self.model, self.train_loader, optimizer, self.criterion,
                self.device, use_amp=False,
            )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"train_loss": float(loss), "train_acc": float(acc)},
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_parameters(parameters)
        self.model.to(self.device)

        metrics = evaluate(
            self.model, self.val_loader, self.criterion,
            self.device, self.classes, use_amp=False,
        )

        return (
            float(metrics["val_loss"]),
            len(self.val_loader.dataset),
            {
                "val_acc": float(metrics["val_acc"]),
                "melanoma_sensitivity": float(metrics["melanoma_sensitivity"]),
                "bcc_sensitivity": float(metrics["bcc_sensitivity"]),
                "composite": float(metrics["composite"]),
            },
        )


def client_fn(context: Context) -> SkinSureClient:
    """Factory function called by Flower to create a client for each hospital.

    Reads configuration from the Flower context:
        - partition-id: which hospital shard (0, 1, 2)
        - backbone: model backbone name
        - data-dir: root of the presplit dataset
        - shard-dir: directory containing hospital shard CSVs
        - local-epochs: training epochs per FL round
        - batch-size: batch size for training
        - lr: learning rate
    """
    import csv
    import shutil
    import tempfile

    cfg = context.run_config
    partition_id = int(context.node_config.get("partition-id", 0))

    backbone = cfg.get("backbone", "efficientnet")
    data_dir = Path(cfg["data-dir"])
    shard_dir = Path(cfg.get("shard-dir", "data/splits/shards/noniid"))
    local_epochs = int(cfg.get("local-epochs", 2))
    batch_size = int(cfg.get("batch-size", 16))
    lr = float(cfg.get("lr", 2e-4))

    hospital_names = ["hospital_a", "hospital_b", "hospital_c"]
    hospital_name = hospital_names[partition_id]
    shard_csv = shard_dir / f"{hospital_name}.csv"

    with open(shard_csv, newline="") as f:
        shard_paths = {row["path"] for row in csv.DictReader(f)}

    work_dir = Path(tempfile.mkdtemp(prefix=f"fl_{hospital_name}_"))

    train_dir = work_dir / "train"
    val_dir = work_dir / "val"

    for path_str in shard_paths:
        src = data_dir / path_str
        if not src.exists():
            continue
        parts = Path(path_str).parts
        original_split = parts[0]
        label = parts[1]

        if original_split == "train":
            dest_root = train_dir
        else:
            dest_root = val_dir

        dest_dir = dest_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(str(src), str(dest))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(backbone, num_classes=5, pretrained=False)
    model = model.to(device)

    _, classes = make_dataloader(
        str(train_dir), get_val_transform(224), batch_size=1, is_train=False,
    )

    return SkinSureClient(
        model=model,
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        device=device,
        classes=classes,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
    )


app = ClientApp(client_fn=client_fn)
