from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ConvNeXtSkinClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.4,
        drop_path_rate: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features  # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def load_convnext_tiny(
    path: str, device: torch.device
) -> tuple[ConvNeXtSkinClassifier, list[str]]:
    """Load a trained ConvNeXt-Tiny checkpoint. Returns (model, classes)."""
    ckpt = torch.load(path, map_location=device)
    model = ConvNeXtSkinClassifier(
        num_classes=len(ckpt["classes"]),
        dropout=ckpt["config"]["dropout"],
        drop_path_rate=ckpt["config"]["drop_path_rate"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    mel = ckpt.get("best_melanoma_sens", float("nan"))
    print(
        f"✓ ConvNeXt-Tiny | val={ckpt['best_val_acc']*100:.2f}% "
        f"test={ckpt['test_acc']*100:.2f}% | mel_sens={mel*100:.1f}%"
    )
    return model, ckpt["classes"]
