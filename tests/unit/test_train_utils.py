"""Tests for src/training/train_utils.py using tiny synthetic models and data."""
from __future__ import annotations

import csv
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.training.train_utils import (
    EXPECTED_CLASSES,
    build_loss_weights,
    build_optimizer,
    build_scheduler,
    create_model,
    evaluate,
    get_train_transform,
    get_val_transform,
    make_dataloader,
    train_one_epoch,
)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a tiny ImageFolder dataset with 5 classes, 4 images each."""
    for cls in EXPECTED_CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(cls_dir / f"{i}.png")
    return str(tmp_path)


class TinyModel(nn.Module):
    """Minimal model for testing the training loop."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class TestBuildLossWeights:
    def test_shape_and_device(self):
        counts = {0: 100, 1: 200, 2: 150, 3: 300, 4: 50}
        w = build_loss_weights(counts, EXPECTED_CLASSES, torch.device("cpu"))
        assert w.shape == (5,)
        assert w.device == torch.device("cpu")

    def test_sums_to_one(self):
        counts = {0: 100, 1: 200, 2: 150, 3: 300, 4: 50}
        w = build_loss_weights(counts, EXPECTED_CLASSES, torch.device("cpu"))
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_clinical_boosts_applied(self):
        counts = {i: 1000 for i in range(5)}
        w = build_loss_weights(counts, EXPECTED_CLASSES, torch.device("cpu"))
        mel_idx = EXPECTED_CLASSES.index("Melanoma")
        ker_idx = EXPECTED_CLASSES.index("Keratosis")
        assert w[mel_idx] > w[ker_idx]


class TestMakeDataloader:
    def test_train_loader(self, synthetic_dataset):
        loader, classes = make_dataloader(
            synthetic_dataset, get_val_transform(32), batch_size=4, is_train=True
        )
        assert set(classes) == set(EXPECTED_CLASSES)
        batch = next(iter(loader))
        assert batch[0].shape[0] <= 4

    def test_val_loader(self, synthetic_dataset):
        loader, classes = make_dataloader(
            synthetic_dataset, get_val_transform(32), batch_size=4, is_train=False
        )
        assert len(classes) == 5


class TestTrainOneEpoch:
    def test_runs_and_returns_metrics(self, synthetic_dataset):
        loader, classes = make_dataloader(
            synthetic_dataset, get_val_transform(32), batch_size=4, is_train=True
        )
        model = TinyModel()
        device = torch.device("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss, acc = train_one_epoch(
            model, loader, optimizer, criterion, device, use_amp=False
        )
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0


class TestEvaluate:
    def test_returns_expected_keys(self, synthetic_dataset):
        loader, classes = make_dataloader(
            synthetic_dataset, get_val_transform(32), batch_size=4, is_train=False
        )
        model = TinyModel()
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, device, classes, use_amp=False)
        assert "val_loss" in metrics
        assert "val_acc" in metrics
        assert "melanoma_sensitivity" in metrics
        assert "bcc_sensitivity" in metrics
        assert "composite" in metrics
        assert "per_class_sensitivity" in metrics

    def test_composite_formula(self, synthetic_dataset):
        loader, classes = make_dataloader(
            synthetic_dataset, get_val_transform(32), batch_size=4, is_train=False
        )
        model = TinyModel()
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, loader, criterion, device, classes, use_amp=False)
        expected = (0.4 * metrics["melanoma_sensitivity"]
                    + 0.3 * metrics["bcc_sensitivity"]
                    + 0.3 * metrics["val_acc"])
        assert abs(metrics["composite"] - expected) < 1e-6


class TestBuildOptimizer:
    def test_frozen_backbone(self):
        model = TinyModel()
        opt = build_optimizer(model, lr=1e-3, weight_decay=1e-4, lr_decay_rate=0.1,
                              freeze_backbone=True)
        assert len(opt.param_groups) == 1
        assert not any(p.requires_grad for p in model.backbone.parameters())

    def test_unfrozen(self):
        model = TinyModel()
        opt = build_optimizer(model, lr=1e-3, weight_decay=1e-4, lr_decay_rate=0.1,
                              freeze_backbone=False)
        assert len(opt.param_groups) == 2


class TestBuildScheduler:
    def test_warmup_increases(self):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = build_scheduler(opt, warmup_epochs=3, num_epochs=10)
        lrs = []
        for _ in range(3):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        assert lrs[0] < lrs[1] < lrs[2]


class TestCreateModel:
    def test_efficientnet(self):
        m = create_model("efficientnet", num_classes=5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_convnext(self):
        m = create_model("convnext", num_classes=5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_swin(self):
        m = create_model("swin", num_classes=5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            create_model("resnet", num_classes=5)
