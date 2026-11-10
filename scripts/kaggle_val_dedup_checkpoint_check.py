"""
Standalone Kaggle script (GPU) -- checkpoint-selection sanity check.

Question: does the composite checkpoint-selection criterion

    S = 0.4 * Sens_mel + 0.3 * Sens_bcc + 0.3 * Acc_val

change its answer when computed on the clean (de-duplicated) validation set
instead of the full, leakage-contaminated validation set?

Background: a pHash/dHash near-duplicate audit (Notebook 2) found 199 of 748
validation images (26.60%) have a near-duplicate (Hamming distance <=5) in
train or test, leaving a clean 549-image validation set
(`val_clean_manifest.csv`). Only ONE checkpoint per backbone exists on disk
(the already-selected v2 retrain) -- so this does not re-run a multi-epoch
selection sweep. It instead runs all three backbones (+ the equal-weight
ensemble) over every image in the full val/ split, then computes S on both
the full set and the clean subset, and reports whether the ranking / the
shipped checkpoint's S changes.

Kaggle setup
------------
1. Add data:
     - jainmanav20/ipd-skin-dataset           (DATA_ROOT below)
     - jainmanav20/checkpoints-new            (CKPT_SRC below)
     - your uploaded results.zip contents, specifically val_clean_manifest.csv
       -- upload it as a new Kaggle dataset (e.g. "val-dedup-results") or paste
       its rows inline; point CLEAN_MANIFEST at wherever it lands.
2. Enable a GPU accelerator (T4 is plenty).
3. Run all cells / run this script top to bottom. Takes minutes on GPU for
   748 images x 3 backbones.

Outputs (to /kaggle/working/):
    val_dedup_logits.csv   -- per-image predictions, all 3 backbones + ensemble
    val_dedup_summary.txt  -- the printed table, also written to disk
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
DATA_ROOT = Path("/kaggle/input/datasets/jainmanav20/ipd-skin-dataset/IPD_Datasets_Final_Split")
CKPT_SRC = Path("/kaggle/input/datasets/jainmanav20/checkpoints-new")
CLEAN_MANIFEST = Path("/kaggle/input/datasets/jainmanav20/val-dedup-results/val_clean_manifest.csv")
OUT_DIR = Path("/kaggle/working")
# ======================================================================

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LOGITS_CACHE = OUT_DIR / "val_dedup_logits.csv"
SUMMARY_FILE = OUT_DIR / "val_dedup_summary.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- models ----
class EfficientSkinClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.35, drop_path_rate=0.15, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b1", pretrained=pretrained, num_classes=0,
            drop_rate=dropout, drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ConvNeXtSkinClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4, drop_path_rate=0.2, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0,
            drop_rate=dropout, drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class SwinSkinClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.4, drop_path_rate=0.2, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_rate=dropout, drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def _resolve(*cands: Path) -> Path:
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"none of these exist: {cands}")


def load_efficientnet_b1(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EfficientSkinClassifier(
        num_classes=len(ckpt["classes"]),
        dropout=ckpt["config"]["dropout"],
        drop_path_rate=ckpt["config"]["drop_path_rate"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"[ok] EfficientNet-B1 | val={ckpt['best_val_acc']*100:.2f}% test={ckpt['test_acc']*100:.2f}%")
    return model, ckpt["classes"]


def load_convnext_tiny(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvNeXtSkinClassifier(
        num_classes=len(ckpt["classes"]),
        dropout=ckpt["config"]["dropout"],
        drop_path_rate=ckpt["config"]["drop_path_rate"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"[ok] ConvNeXt-Tiny | val={ckpt['best_val_acc']*100:.2f}% test={ckpt['test_acc']*100:.2f}%")
    return model, ckpt["classes"]


def load_swin_tiny(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SwinSkinClassifier(
        model_name=ckpt["config"]["model_name"],
        num_classes=len(ckpt["classes"]),
        dropout=ckpt["config"]["dropout"],
        drop_path_rate=ckpt["config"]["drop_path_rate"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"[ok] Swin-Tiny | val={ckpt['best_val_acc']*100:.2f}% test={ckpt['test_acc']*100:.2f}%")
    return model, ckpt["classes"]


# ----------------------------------------------------------- preprocessing --
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _transform(image).unsqueeze(0)


# --------------------------------------------------------------- manifest ---
def kaggle_path_to_filename(kaggle_path: str) -> tuple[str, str]:
    parts = kaggle_path.strip().split("/")
    idx = parts.index("val")
    cls = parts[idx + 1]
    fname = parts[idx + 2]
    return cls, fname


def load_clean_filenames() -> set[tuple[str, str]]:
    clean = set()
    with open(CLEAN_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls, fname = kaggle_path_to_filename(row["path"])
            clean.add((cls, fname))
    return clean


def collect_val_images() -> list[tuple[str, Path]]:
    items = []
    for cls in CLASSES:
        cls_dir = DATA_ROOT / "val" / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing val class dir: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((cls, p))
    return items


# ---------------------------------------------------------------- metrics ---
def compute_metrics(rows: list[dict], pred_key: str) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r[pred_key] == r["true_class"])
    acc = correct / n if n else float("nan")

    mel_rows = [r for r in rows if r["true_class"] == "Melanoma"]
    sens_mel = (
        sum(1 for r in mel_rows if r[pred_key] == "Melanoma") / len(mel_rows)
        if mel_rows else float("nan")
    )

    bcc_rows = [r for r in rows if r["true_class"] == "Basal_Cell_Carcinoma"]
    sens_bcc = (
        sum(1 for r in bcc_rows if r[pred_key] == "Basal_Cell_Carcinoma") / len(bcc_rows)
        if bcc_rows else float("nan")
    )

    composite = 0.4 * sens_mel + 0.3 * sens_bcc + 0.3 * acc
    return {
        "n": n, "n_mel": len(mel_rows), "n_bcc": len(bcc_rows),
        "acc": acc, "sens_mel": sens_mel, "sens_bcc": sens_bcc, "S": composite,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")

    ckpt_eff = _resolve(
        CKPT_SRC / "efficientnet" / "efficientnet_b1_5cls_skin_v2.pth",
        CKPT_SRC / "efficientNet" / "efficientnet_b1_5cls_skin_v2.pth",
        CKPT_SRC / "efficientnet_b1_5cls_skin_v2.pth",
    )
    ckpt_cnxt = _resolve(
        CKPT_SRC / "convnext" / "convnext_tiny_5cls_skin_v2.pth",
        CKPT_SRC / "convNext" / "convnext_tiny_5cls_skin_v2.pth",
        CKPT_SRC / "convnext_tiny_5cls_skin_v2.pth",
    )
    ckpt_swin = _resolve(
        CKPT_SRC / "swin" / "swin_tiny_5cls_skin_v2.pth",
        CKPT_SRC / "swin_tiny_5cls_skin_v2.pth",
    )

    print("Loading checkpoints...")
    eff_model, eff_classes = load_efficientnet_b1(ckpt_eff, DEVICE)
    cnxt_model, cnxt_classes = load_convnext_tiny(ckpt_cnxt, DEVICE)
    swin_model, swin_classes = load_swin_tiny(ckpt_swin, DEVICE)
    assert eff_classes == cnxt_classes == swin_classes == CLASSES, (
        eff_classes, cnxt_classes, swin_classes,
    )

    # Equal weights per configs/models.yaml (1.0/1.0/1.0 -> averages to 1/3 each).
    w_eff = w_cnxt = w_swin = 1.0 / 3.0

    items = collect_val_images()
    print(f"Found {len(items)} val images across 5 target classes (expect 748)")

    rows = []
    for i, (true_cls, img_path) in enumerate(items):
        img = Image.open(img_path)
        x = preprocess_image(img).to(DEVICE)

        with torch.no_grad():
            p_eff = F.softmax(eff_model(x), dim=1).squeeze(0)
            p_cnxt = F.softmax(cnxt_model(x), dim=1).squeeze(0)
            p_swin = F.softmax(swin_model(x), dim=1).squeeze(0)
            p_ens = w_eff * p_eff + w_cnxt * p_cnxt + w_swin * p_swin

        rows.append({
            "filename": img_path.name,
            "true_class": true_cls,
            "pred_eff": CLASSES[p_eff.argmax().item()],
            "pred_cnxt": CLASSES[p_cnxt.argmax().item()],
            "pred_swin": CLASSES[p_swin.argmax().item()],
            "pred_ens": CLASSES[p_ens.argmax().item()],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}")

    with open(LOGITS_CACHE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cached {len(rows)} rows to {LOGITS_CACHE}")

    clean_set = load_clean_filenames()
    clean_rows = [r for r in rows if (r["true_class"], r["filename"]) in clean_set]

    lines = []
    lines.append(f"Full val set: {len(rows)} images")
    lines.append(f"Clean val set (matched to manifest): {len(clean_rows)} images "
                 f"(manifest has {len(clean_set)} entries)")

    model_keys = {
        "EfficientNet-B1": "pred_eff",
        "ConvNeXt-Tiny": "pred_cnxt",
        "Swin-Tiny": "pred_swin",
        "Ensemble": "pred_ens",
    }

    lines.append("=" * 100)
    lines.append(f"{'Backbone':<18}{'Set':<14}{'n':>5}{'Acc_val':>10}{'Sens_mel':>10}{'Sens_bcc':>10}{'S':>10}")
    lines.append("-" * 100)

    results = {}
    for name, key in model_keys.items():
        full_m = compute_metrics(rows, key)
        clean_m = compute_metrics(clean_rows, key)
        results[name] = {"full": full_m, "clean": clean_m}
        for label, m in (("full(748)", full_m), ("clean(dedup)", clean_m)):
            lines.append(f"{name:<18}{label:<14}{m['n']:>5}{m['acc']*100:>9.2f}%{m['sens_mel']*100:>9.2f}%"
                          f"{m['sens_bcc']*100:>9.2f}%{m['S']:>10.4f}")
        lines.append("-" * 100)

    full_ranking = sorted(
        [n for n in model_keys if n != "Ensemble"],
        key=lambda n: results[n]["full"]["S"], reverse=True,
    )
    clean_ranking = sorted(
        [n for n in model_keys if n != "Ensemble"],
        key=lambda n: results[n]["clean"]["S"], reverse=True,
    )
    lines.append(f"\nBackbone ranking by S, full val:  {full_ranking}")
    lines.append(f"Backbone ranking by S, clean val: {clean_ranking}")
    lines.append(f"Ranking changes: {full_ranking != clean_ranking}")

    lines.append("\nShipped checkpoint S deltas (clean - full):")
    for name in model_keys:
        d = results[name]["clean"]["S"] - results[name]["full"]["S"]
        lines.append(f"  {name:<18} full S={results[name]['full']['S']:.4f}  "
                      f"clean S={results[name]['clean']['S']:.4f}  delta={d:+.4f}")

    report = "\n".join(lines)
    print(report)
    SUMMARY_FILE.write_text(report, encoding="utf-8")
    print(f"\nWrote {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
