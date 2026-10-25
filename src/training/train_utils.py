"""Reusable training and evaluation utilities extracted from the Colab notebooks.

Used by both centralized training and federated learning clients.
See fedler.md Phase 0 and src/models/CLAUDE.md for design decisions.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXPECTED_CLASSES = [
    "Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"
]

CLINICAL_BOOSTS = {
    "Melanoma": 2.0,
    "Basal_Cell_Carcinoma": 2.0,
    "Tinea": 3.0,
    "Eczema": 1.5,
}

CLINICAL_TARGETS = {
    "Melanoma": 0.95,
    "Basal_Cell_Carcinoma": 0.90,
    "Tinea": 0.80,
    "Eczema": 0.80,
    "Keratosis": 0.80,
}

DEFAULT_CFG = dict(
    img_size=224,
    batch_size=16,
    num_epochs=30,
    lr=2e-4,
    weight_decay=2e-4,
    label_smooth=0.1,
    warmup_epochs=3,
    use_amp=True,
    num_workers=1,
    freeze_epochs=6,
    patience=6,
    lr_decay_rate=0.1,
    grad_clip=1.0,
)


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loss_weights(class_counts: dict[int, int], classes: list[str], device: torch.device) -> torch.Tensor:
    """Inverse-sqrt class weights with clinical boosts (Melanoma 2x, BCC 2x, Tinea 3x, Eczema 1.5x)."""
    num_cls = len(classes)
    counts = np.array([class_counts[i] for i in range(num_cls)], dtype=np.float32)
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum()

    for cls_name, boost in CLINICAL_BOOSTS.items():
        if cls_name in classes:
            weights[classes.index(cls_name)] *= boost
    weights /= weights.sum()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_dataloader(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 1,
    is_train: bool = False,
) -> tuple[DataLoader, list[str]]:
    """Build a DataLoader from an ImageFolder directory.

    For training, uses WeightedRandomSampler to balance classes.
    Returns (dataloader, class_names).
    """
    dataset = ImageFolder(data_dir, transform=transform)
    classes = dataset.classes
    found = set(classes)
    expected = set(EXPECTED_CLASSES)
    if found != expected:
        raise ValueError(f"Class mismatch: expected {expected}, found {found}")

    if is_train:
        class_counts = Counter(dataset.targets)
        sample_weights = 1.0 / torch.tensor(
            [class_counts[i] for i in range(len(classes))], dtype=torch.float
        )
        weights = [sample_weights[t] for t in dataset.targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return loader, classes


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, lbls in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, lbls)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == lbls).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: list[str],
    use_amp: bool = True,
) -> dict[str, Any]:
    """Evaluate model on a dataloader. Returns metrics dict.

    Keys: val_loss, val_acc, melanoma_sensitivity, bcc_sensitivity, composite,
          per_class_sensitivity (dict), per_class_specificity (dict).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, lbls in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, lbls)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == lbls).sum().item()
        total += imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_loss = total_loss / total
    val_acc = correct / total

    per_class_sens = {}
    per_class_spec = {}
    for i, cls in enumerate(classes):
        tp = ((all_preds == i) & (all_labels == i)).sum()
        fn = ((all_preds != i) & (all_labels == i)).sum()
        tn = ((all_preds != i) & (all_labels != i)).sum()
        fp = ((all_preds == i) & (all_labels != i)).sum()
        per_class_sens[cls] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class_spec[cls] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mel_sens = per_class_sens.get("Melanoma", 0.0)
    bcc_sens = per_class_sens.get("Basal_Cell_Carcinoma", 0.0)
    composite = 0.4 * mel_sens + 0.3 * bcc_sens + 0.3 * val_acc

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "melanoma_sensitivity": mel_sens,
        "bcc_sensitivity": bcc_sens,
        "composite": composite,
        "per_class_sensitivity": per_class_sens,
        "per_class_specificity": per_class_spec,
    }


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    lr_decay_rate: float,
    freeze_backbone: bool = False,
) -> optim.Optimizer:
    """AdamW with layer-wise LR: backbone at lr * lr_decay_rate, head at lr."""
    if freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False

    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n and p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * lr_decay_rate})
    param_groups.append({"params": head_params, "lr": lr})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    num_epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_model(backbone: str, num_classes: int = 5, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    """Create a model by backbone name. Supports efficientnet, convnext, swin."""
    if backbone == "efficientnet":
        from src.models.efficientnet import EfficientSkinClassifier
        return EfficientSkinClassifier(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.35),
            drop_path_rate=kwargs.get("drop_path_rate", 0.15),
            pretrained=pretrained,
        )
    elif backbone == "convnext":
        from src.models.convnext import ConvNeXtSkinClassifier
        return ConvNeXtSkinClassifier(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.4),
            drop_path_rate=kwargs.get("drop_path_rate", 0.2),
            pretrained=pretrained,
        )
    elif backbone == "swin":
        from src.models.swin import SwinSkinClassifier
        return SwinSkinClassifier(
            model_name=kwargs.get("model_name", "swin_tiny_patch4_window7_224"),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.4),
            drop_path_rate=kwargs.get("drop_path_rate", 0.2),
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from: efficientnet, convnext, swin")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: list[str],
    device: torch.device,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Full training loop with freeze-then-finetune, early stopping, and composite checkpointing.

    Returns dict with best_composite, best_val_acc, best_melanoma_sens, history, best_state_dict.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    class_counts = Counter(train_loader.dataset.targets)
    weights_tensor = build_loss_weights(class_counts, classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=cfg["label_smooth"])

    use_amp = cfg["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"], cfg["lr_decay_rate"],
                                freeze_backbone=True)
    scheduler = build_scheduler(optimizer, cfg["warmup_epochs"], cfg["num_epochs"])

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        "melanoma_sensitivity": [], "bcc_sensitivity": [], "composite": [],
    }
    best_composite = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, cfg["num_epochs"] + 1):
        if epoch == cfg["freeze_epochs"] + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"], cfg["lr_decay_rate"])
            scheduler = build_scheduler(optimizer, cfg["warmup_epochs"], cfg["num_epochs"])

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp, grad_clip=cfg["grad_clip"],
        )
        metrics = evaluate(model, val_loader, criterion, device, classes, use_amp=use_amp)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(metrics["val_acc"])
        history["melanoma_sensitivity"].append(metrics["melanoma_sensitivity"])
        history["bcc_sensitivity"].append(metrics["bcc_sensitivity"])
        history["composite"].append(metrics["composite"])

        if metrics["composite"] > best_composite:
            best_composite = metrics["composite"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(
            f"  Ep {epoch:>3}/{cfg['num_epochs']} | "
            f"loss {tr_loss:.4f}/{metrics['val_loss']:.4f} | "
            f"acc {tr_acc*100:.1f}/{metrics['val_acc']*100:.1f}% | "
            f"mel {metrics['melanoma_sensitivity']*100:.1f}% | "
            f"comp {metrics['composite']:.4f}"
            f"{' *' if patience_cnt == 0 else ''}"
        )

        if patience_cnt >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch}.")
            break

    return {
        "best_composite": best_composite,
        "best_val_acc": max(history["val_acc"]),
        "best_melanoma_sens": max(history["melanoma_sensitivity"]),
        "history": history,
        "best_state_dict": best_state,
    }
