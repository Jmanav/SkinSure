"""
Checkpoint-selection sanity check: does the composite criterion

    S = 0.4 * Sens_mel + 0.3 * Sens_bcc + 0.3 * Acc_val

change its answer when computed on the clean (de-duplicated) validation set
instead of the full, leakage-contaminated validation set?

Only one checkpoint per backbone exists on disk (the already-selected v2
retrain -- see models/checkpoints/{efficientNet,convNext,swin}/*_v2.pth), so
this script does NOT re-run a multi-epoch selection sweep. It instead:

  1. Runs all three backbones (+ the equal-weight ensemble) over every image
     in the full local val/ split (748 images across the 5 target classes),
     caching per-image logits/predictions to .nb1out/val_dedup_logits.csv.
  2. Computes Sens_mel, Sens_bcc, Acc_val, and S on:
       (a) the full 748-image val set
       (b) the clean subset defined by .nb1out/val_dedup_results/val_clean_manifest.csv
  3. Reports whether the ranking / the shipped checkpoint's S changes.

Usage:
    ./venv/Scripts/python.exe scripts/val_dedup_checkpoint_check.py [--recompute]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.convnext import load_convnext_tiny
from src.models.efficientnet import load_efficientnet_b1
from src.models.swin import load_swin_tiny
from src.preprocessing.inference_transform import preprocess_image
from src.utils.config import get_checkpoint_paths, get_model_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_ROOT = PROJECT_ROOT / "data/raw/IPD_Datasets_Final_Split/val"
CLEAN_MANIFEST = PROJECT_ROOT / ".nb1out/val_dedup_results/val_clean_manifest.csv"
LOGITS_CACHE = PROJECT_ROOT / ".nb1out/val_dedup_logits.csv"

TARGET_CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]


def kaggle_path_to_filename(kaggle_path: str) -> tuple[str, str]:
    """Map a Kaggle-style manifest path to (class_name, filename)."""
    parts = kaggle_path.strip().split("/")
    # .../val/<Class>/<filename>
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
    """All (class, path) pairs in the local val/ split, restricted to the 5 target classes."""
    items = []
    for cls in TARGET_CLASSES:
        cls_dir = VAL_ROOT / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing val class dir: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((cls, p))
    return items


def run_inference_and_cache() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_checkpoint_paths()

    print("Loading checkpoints...")
    eff_model, eff_classes = load_efficientnet_b1(str(paths.efficientnet), device)
    cnxt_model, cnxt_classes = load_convnext_tiny(str(paths.convnext), device)
    swin_model, swin_classes = load_swin_tiny(str(paths.swin), device)
    assert eff_classes == cnxt_classes == swin_classes == TARGET_CLASSES, (
        eff_classes, cnxt_classes, swin_classes,
    )

    model_cfg = get_model_config()
    weights = model_cfg.weights  # dict: {efficientnet, convnext, swin} per configs/models.yaml
    total_w = sum(weights.values())
    w_eff = weights["efficientnet"] / total_w
    w_cnxt = weights["convnext"] / total_w
    w_swin = weights["swin"] / total_w
    print(f"Ensemble weights: eff={w_eff:.3f} cnxt={w_cnxt:.3f} swin={w_swin:.3f}")

    items = collect_val_images()
    print(f"Found {len(items)} val images across 5 target classes")

    rows = []
    for i, (true_cls, img_path) in enumerate(items):
        img = Image.open(img_path).convert("RGB")
        x = preprocess_image(img).to(device)

        with torch.no_grad():
            p_eff = F.softmax(eff_model(x), dim=1).squeeze(0)
            p_cnxt = F.softmax(cnxt_model(x), dim=1).squeeze(0)
            p_swin = F.softmax(swin_model(x), dim=1).squeeze(0)
            p_ens = w_eff * p_eff + w_cnxt * p_cnxt + w_swin * p_swin

        pred_eff = TARGET_CLASSES[p_eff.argmax().item()]
        pred_cnxt = TARGET_CLASSES[p_cnxt.argmax().item()]
        pred_swin = TARGET_CLASSES[p_swin.argmax().item()]
        pred_ens = TARGET_CLASSES[p_ens.argmax().item()]

        rows.append({
            "filename": img_path.name,
            "true_class": true_cls,
            "pred_eff": pred_eff,
            "pred_cnxt": pred_cnxt,
            "pred_swin": pred_swin,
            "pred_ens": pred_ens,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}")

    LOGITS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGITS_CACHE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cached {len(rows)} rows to {LOGITS_CACHE}")


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
        "n": n,
        "n_mel": len(mel_rows),
        "n_bcc": len(bcc_rows),
        "acc": acc,
        "sens_mel": sens_mel,
        "sens_bcc": sens_bcc,
        "S": composite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Force re-run inference even if cache exists")
    args = parser.parse_args()

    if args.recompute or not LOGITS_CACHE.exists():
        run_inference_and_cache()
    else:
        print(f"Using cached predictions from {LOGITS_CACHE} (pass --recompute to redo)")

    with open(LOGITS_CACHE, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    clean_set = load_clean_filenames()
    clean_rows = [r for r in all_rows if (r["true_class"], r["filename"]) in clean_set]

    print(f"\nFull val set: {len(all_rows)} images")
    print(f"Clean val set (matched to manifest): {len(clean_rows)} images "
          f"(manifest has {len(clean_set)} entries)")

    model_keys = {
        "EfficientNet-B1": "pred_eff",
        "ConvNeXt-Tiny": "pred_cnxt",
        "Swin-Tiny": "pred_swin",
        "Ensemble": "pred_ens",
    }

    print("\n" + "=" * 100)
    header = f"{'Backbone':<18}{'Set':<10}{'n':>5}{'Acc_val':>10}{'Sens_mel':>10}{'Sens_bcc':>10}{'S':>10}"
    print(header)
    print("-" * 100)

    results = {}
    for name, key in model_keys.items():
        full_m = compute_metrics(all_rows, key)
        clean_m = compute_metrics(clean_rows, key)
        results[name] = {"full": full_m, "clean": clean_m}
        for label, m in (("full(748)", full_m), ("clean(dedup)", clean_m)):
            print(f"{name:<18}{label:<10}{m['n']:>5}{m['acc']*100:>9.2f}%{m['sens_mel']*100:>9.2f}%"
                  f"{m['sens_bcc']*100:>9.2f}%{m['S']:>10.4f}")
        print("-" * 100)

    # Ranking comparison
    full_ranking = sorted(
        [n for n in model_keys if n != "Ensemble"],
        key=lambda n: results[n]["full"]["S"],
        reverse=True,
    )
    clean_ranking = sorted(
        [n for n in model_keys if n != "Ensemble"],
        key=lambda n: results[n]["clean"]["S"],
        reverse=True,
    )
    print(f"\nBackbone ranking by S, full val:  {full_ranking}")
    print(f"Backbone ranking by S, clean val: {clean_ranking}")
    print(f"Ranking changes: {full_ranking != clean_ranking}")

    print("\nShipped checkpoint S deltas (clean - full):")
    for name in model_keys:
        d = results[name]["clean"]["S"] - results[name]["full"]["S"]
        print(f"  {name:<18} full S={results[name]['full']['S']:.4f}  "
              f"clean S={results[name]['clean']['S']:.4f}  delta={d:+.4f}")


if __name__ == "__main__":
    main()
