"""Notebook 7 — Grad-CAM faithfulness: deletion/AOPC vs structure-matched random baselines.

RUN THIS AS A KAGGLE NOTEBOOK ON A T4. Paste each numbered section into its own cell,
or run the whole file as a single script cell — it is written to work either way.

This is the first faithfulness measurement of SkinSure's explanation layer
(CLAUDE.md gap 10). Grad-CAM is on `/predict` for latency and output-shape reasons,
never because it was shown to explain better than anything else. This measures it.

SELF-CONTAINED by design, following the rules in paper/experiments/README.md: the
model classes, the eval transform, Grad-CAM, the deletion curve and the statistics are
all copied verbatim from the repo so nothing is imported from `src/`. Outputs go to
/kaggle/working/.

--------------------------------------------------------------------------------------
WHAT THE THREE ARMS ARE, AND WHY TWO OF THEM ARE NOT INTERCHANGEABLE
--------------------------------------------------------------------------------------
  gradcam      Weighted ensemble Grad-CAM (7x7 per backbone, bilinear to 224).

  random7x7    Uniform noise on a 7x7 grid, upsampled with the SAME bilinear call
               Grad-CAM uses. 7x7 is Grad-CAM's NATIVE resolution in all three
               backbones at 224px, so this baseline produces maps with the same
               spatial structure — smooth, blobby, low-frequency. Deleting the top-k%
               removes contiguous regions of comparable size in both arms, so the ONLY
               difference is whether the blobs land on informative pixels.
               *** THIS IS THE FAIR COMPARATOR. `gradcam - random7x7` IS THE RESULT. ***

  randompixel  Uniform i.i.d. noise at full 224x224. Deleting its top-k% removes a
               fine speckle scattered across the whole frame, which shreds local
               texture everywhere at once and pushes the image far off the training
               distribution. p(class) then collapses REGARDLESS of where the deletions
               land. It therefore posts the HIGHEST AOPC while explaining NOTHING.
               Treat it as a sanity check that AOPC is sensitive to out-of-distribution
               damage — NOT as a rival method. Do not report it as a winner.

--------------------------------------------------------------------------------------
KAGGLE DATASETS TO ATTACH (same three NB1 uses)
--------------------------------------------------------------------------------------
  DATA_ROOT       jainmanav20/ipd-skin-dataset   -> IPD_Datasets_Final_Split/{split}/<Class>/
  CKPT_SRC        jainmanav20/checkpoints-new    -> the three *_5cls_skin_v2.pth
  CLEAN_MANIFEST  jainmanav20/dedup-eval-csv     -> test_clean_manifest.csv (the 565 set)

Expected runtime on a T4: roughly 1.5-4 s per image for all 3 arms x 3 seeds, so
~5-12 minutes for the default 30/class = 150 images. On the 2-core dev CPU the same
work ran at ~20 s per image for Grad-CAM alone, which is why this moved to Kaggle.
"""

# =====================================================================================
# 1. CONFIG — EDIT THESE
# =====================================================================================
DATA_ROOT      = '/kaggle/input/datasets/jainmanav20/ipd-skin-dataset/IPD_Datasets_Final_Split'
CKPT_SRC       = '/kaggle/input/datasets/jainmanav20/checkpoints-new'
CLEAN_MANIFEST = '/kaggle/input/datasets/jainmanav20/dedup-eval-csv/test_clean_manifest.csv'
OUT_DIR        = '/kaggle/working'

PER_CLASS    = 30          # images per class; 150 total. Lower to 15 for a faster pass.
SAMPLE_SEED  = 20261102    # the seed the local run used; keep it to reuse that sample
RANDOM_SEEDS = 3           # independent draws per random baseline, averaged before pairing
SEED         = 42          # global seed (paper/experiments convention)

CLASSES = ['Basal_Cell_Carcinoma', 'Eczema', 'Keratosis', 'Melanoma', 'Tinea']
# Equal weights from configs/models.yaml, NORMALISED to sum to 1.
# SkinEnsemble does `w / sum(weights)` in its constructor, so the raw (1, 1, 1) in the
# config means (1/3, 1/3, 1/3) at inference. Using the raw values here instead makes
# every "probability" 3x too large -- p(class) came out at 2.6 in the first Kaggle run.
# Argmax and AOPC are both scale-invariant/linear, so that bug changed no ranking, no
# win count and no p-value, only the absolute magnitudes. Normalise anyway: a number
# printed as a probability must be one.
_RAW_WEIGHTS = (1.0, 1.0, 1.0)
ENSEMBLE_WEIGHTS = tuple(w / sum(_RAW_WEIGHTS) for w in _RAW_WEIGHTS)
DELETION_PERCENTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
CAM_GRID = 7               # Grad-CAM's native resolution at 224px input

import csv
import json
import math
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f'device: {DEVICE}  torch {torch.__version__}  timm {timm.__version__}')
if DEVICE.type == 'cuda':
    print(f'gpu: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: no GPU. This will be slow — switch the accelerator to T4 x1.')


# =====================================================================================
# 2. REASSEMBLE CHECKPOINTS (Kaggle explodes .pth on upload) — verbatim from NB1
# =====================================================================================
CKPT_SRC = Path(CKPT_SRC)
CKPT_BUILD = Path('/kaggle/working/ckpts')

# The retrains' stored values, so a silently damaged rebuild is caught here rather than
# six cells later. auc is held to 1e-4: ConvNeXt's console output reported 4 dp only.
EXPECTED = {
    'efficientnet_b1_5cls_skin_v2': (0.8473177441540578, 0.9704132075327238),
    'swin_tiny_5cls_skin_v2':       (0.8596973865199450, 0.9737358109312158),
    'convnext_tiny_5cls_skin_v2':   (0.8913342503000000, 0.9684000000000000),
}


def _rebuild(exploded: Path, out: Path) -> Path:
    """Re-zip an exploded torch checkpoint. Kaggle nests the payload one level deeper
    under a repeat of the folder name; torch.load needs every member under a single
    top-level folder and the archive uncompressed (ZIP_STORED)."""
    inner = exploded / exploded.name
    base = inner if inner.is_dir() else exploded
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
        for f in sorted(base.rglob('*')):
            if f.is_file():
                z.write(f, f'{base.name}/{f.relative_to(base)}')
    return out


if not CKPT_SRC.exists():
    raise FileNotFoundError(f'checkpoint dataset not found: {CKPT_SRC}\n'
                            'Fix the slug — it is the dataset name in the file browser.')

real = sorted(p for p in CKPT_SRC.rglob('*_5cls_skin_v2.pth') if p.is_file())
if real:
    CKPT_BUILD = real[0].parent
    print(f'found {len(real)} real .pth files; no rebuild needed')
else:
    CKPT_BUILD.mkdir(parents=True, exist_ok=True)
    exploded = sorted(p for p in CKPT_SRC.iterdir() if p.is_dir() and p.name in EXPECTED)
    if len(exploded) != 3:
        raise FileNotFoundError(
            f'expected 3 exploded checkpoint folders in {CKPT_SRC}, '
            f'found {[p.name for p in exploded]}.\n'
            f'directory holds: {sorted(p.name for p in CKPT_SRC.iterdir())[:25]}')
    for d in exploded:
        out = _rebuild(d, CKPT_BUILD / f'{d.name}.pth')
        ck = torch.load(out, map_location='cpu', weights_only=False)
        acc, auc = float(ck['test_acc']), float(ck['macro_auc'])
        e_acc, e_auc = EXPECTED[d.name]
        if abs(acc - e_acc) > 1e-9 or abs(auc - e_auc) > 1e-4:
            raise RuntimeError(f'{d.name}: rebuilt checkpoint does not match stored '
                               f'metrics (acc {acc} vs {e_acc}, auc {auc} vs {e_auc})')
        print(f'  rebuilt {d.name}.pth  acc={acc*100:.2f}%  auc={auc:.4f}  OK')
print(f'CKPT_BUILD = {CKPT_BUILD}')


# =====================================================================================
# 3. MODEL CLASSES — copied verbatim from src/models/*.py so strict=True loads
# =====================================================================================
class EfficientSkinClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.35, drop_path_rate=0.15, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b1', pretrained=pretrained,
                                          num_classes=0, drop_rate=dropout,
                                          drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features          # 1280
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ConvNeXtSkinClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4, drop_path_rate=0.2, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained,
                                          num_classes=0, drop_rate=dropout,
                                          drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features          # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class SwinSkinClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.4, drop_path_rate=0.2,
                 pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, drop_rate=dropout,
                                          drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features          # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


def _find_ckpt(stem: str) -> Path:
    hits = sorted(p for p in Path(CKPT_BUILD).rglob(f'{stem}*.pth') if p.is_file())
    if not hits:
        raise FileNotFoundError(f'no checkpoint matching {stem}*.pth under {CKPT_BUILD}')
    return hits[0]


def load_all():
    """Load the three v2 backbones. Returns (backbones dict, classes)."""
    out, classes = {}, None
    specs = [
        ('efficientnet', 'efficientnet_b1_5cls_skin_v2', EfficientSkinClassifier),
        ('convnext',     'convnext_tiny_5cls_skin_v2',   ConvNeXtSkinClassifier),
        ('swin',         'swin_tiny_5cls_skin_v2',       SwinSkinClassifier),
    ]
    for name, stem, cls in specs:
        ck = torch.load(_find_ckpt(stem), map_location=DEVICE, weights_only=False)
        cfg = ck['config']
        if name == 'swin':
            model = cls(model_name=cfg['model_name'], num_classes=len(ck['classes']),
                        dropout=cfg['dropout'], drop_path_rate=cfg['drop_path_rate'])
        else:
            model = cls(num_classes=len(ck['classes']), dropout=cfg['dropout'],
                        drop_path_rate=cfg['drop_path_rate'])
        model.load_state_dict(ck['model_state'])          # strict=True by default
        out[name] = model.to(DEVICE).eval()
        if classes is None:
            classes = list(ck['classes'])
        elif list(ck['classes']) != classes:
            raise RuntimeError(f'{name} class order differs: {ck["classes"]} vs {classes}')
        print(f'  [ok] {name:13s} test={ck["test_acc"]*100:.2f}% auc={ck["macro_auc"]:.4f}')
    if classes != CLASSES:
        raise RuntimeError(f'checkpoint classes {classes} != expected {CLASSES}')
    return out, classes


BACKBONES, CKPT_CLASSES = load_all()
print(f'classes: {CKPT_CLASSES}')


# =====================================================================================
# 4. PREPROCESSING / GRAD-CAM / DELETION CURVE — verbatim from src/
# =====================================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return _transform(image).unsqueeze(0)


# The last feature map before global pooling in each backbone (timm 1.0.x, 224px):
#   efficientnet_b1 : backbone.bn2        -> (B, 1280, 7, 7)
#   convnext_tiny   : backbone.stages[-1] -> (B,  768, 7, 7)
#   swin_tiny       : backbone.layers[-1] -> (B, 7, 7, 768)   <- NHWC
TARGETS = {
    'efficientnet': (lambda m: m.backbone.bn2, False),
    'convnext':     (lambda m: m.backbone.stages[-1], False),
    'swin':         (lambda m: m.backbone.layers[-1], True),
}


def _minmax(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    peak = cam.max()
    return cam / peak if peak > 0 else np.zeros_like(cam)


def grad_cam(model, image_tensor, class_idx, layer, channels_last=False) -> np.ndarray:
    """Grad-CAM for one image and one class, scaled to [0,1]. Returns (H, W)."""
    captured = {}

    def hook(_m, _i, output):
        captured['act'] = output

    handle = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = image_tensor.detach().to(DEVICE).clone().requires_grad_(True)
            logits = model(x)
            if 'act' not in captured:
                raise RuntimeError('target layer did not run during the forward pass')
            act = captured['act']
            grad = torch.autograd.grad(logits[0, class_idx], act)[0]
    finally:
        handle.remove()

    if channels_last:                                  # Swin is NHWC
        act, grad = act.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
    weights = grad.mean(dim=(2, 3), keepdim=True)      # one weight per channel
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))   # positive evidence only
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode='bilinear',
                        align_corners=False)
    return _minmax(cam[0, 0].detach().cpu().numpy())


def ensemble_heatmap(image_tensor, class_idx) -> np.ndarray:
    """Weighted average of the three backbones' Grad-CAM maps, each scaled to [0,1]
    first so no backbone dominates through raw gradient scale."""
    maps = []
    for (name, model), weight in zip(BACKBONES.items(), ENSEMBLE_WEIGHTS):
        getter, chlast = TARGETS[name]
        maps.append(weight * grad_cam(model, image_tensor, class_idx,
                                      getter(model), chlast))
    return _minmax(np.sum(maps, axis=0))


@torch.no_grad()
def ensemble_probs(image_tensor) -> np.ndarray:
    """Weighted softmax average — what SkinEnsemble.predict uses."""
    x = image_tensor.to(DEVICE)
    total = 0.0
    for model, weight in zip(BACKBONES.values(), ENSEMBLE_WEIGHTS):
        total = total + weight * torch.softmax(model(x), dim=1)
    return total[0].cpu().numpy()


def deletion_curve_reference(image_tensor, heatmap, class_idx,
                             percents=DELETION_PERCENTS):
    """One forward per deletion level. The reference implementation; used to verify
    the batched version on the first image."""
    height, width = heatmap.shape
    order = np.argsort(-np.abs(heatmap).ravel(), kind='stable')
    curve = []
    for pct in percents:
        k = int(order.size * pct / 100)
        modified = image_tensor.clone()
        if k:
            rows, cols = np.divmod(order[:k], width)
            modified[:, :, rows, cols] = 0.0           # 0.0 = ImageNet mean, not black
        curve.append(float(ensemble_probs(modified)[class_idx]))
    return curve


def deletion_curve_batched(image_tensor, heatmap, class_idx,
                           percents=DELETION_PERCENTS):
    """Identical numbers, one batched forward. All 11 levels are independent, so they
    stack into a single batch — this is where the GPU win comes from."""
    height, width = heatmap.shape
    order = np.argsort(-np.abs(heatmap).ravel(), kind='stable')
    variants = []
    for pct in percents:
        k = int(order.size * pct / 100)
        modified = image_tensor.clone()
        if k:
            rows, cols = np.divmod(order[:k], width)
            modified[:, :, rows, cols] = 0.0
        variants.append(modified)
    batch = torch.cat(variants, dim=0).to(DEVICE)
    with torch.no_grad():
        total = 0.0
        for model, weight in zip(BACKBONES.values(), ENSEMBLE_WEIGHTS):
            total = total + weight * torch.softmax(model(batch), dim=1)
    return [float(v) for v in total[:, class_idx].cpu().numpy()]


def aopc(curve) -> float:
    """Mean drop from the undeleted confidence. Higher = found pixels that matter."""
    return float(np.mean([curve[0] - c for c in curve[1:]]))


def random_map(rng: np.random.Generator, grid: int) -> np.ndarray:
    """Uniform noise on a grid x grid map, bilinearly upsampled to 224x224 using the
    SAME interpolate call grad_cam uses, so random7x7 matches Grad-CAM's granularity."""
    if grid >= 224:
        return rng.random((224, 224))
    small = torch.from_numpy(rng.random((1, 1, grid, grid)).astype(np.float32))
    up = F.interpolate(small, size=(224, 224), mode='bilinear', align_corners=False)
    return up[0, 0].numpy()


# =====================================================================================
# 5. SAMPLE — stratified draw from the de-duplicated 565-image clean test set
# =====================================================================================
def load_clean_set():
    """Group the clean-set test images by true class.

    Prefers the 565-image de-duplicated manifest from Notebook 2. Falls back to the
    full 727-image test folder, which is CONTAMINATED (162 images have a pHash
    near-duplicate in train/val) — the fallback shouts about it, because the two bases
    must never be compared with each other.
    """
    by_class = {c: [] for c in CLASSES}
    man = Path(CLEAN_MANIFEST) if CLEAN_MANIFEST else None
    if man and man.exists():
        with man.open(encoding='utf-8-sig') as fh:
            rows = list(csv.DictReader(fh))
        pathcol = next((c for c in ('path', 'filepath', 'file', 'filename')
                        if c in rows[0]), None)
        labelcol = next((c for c in ('label', 'true_class', 'class') if c in rows[0]),
                        None)
        if pathcol is None:
            raise RuntimeError(f'manifest has no path-like column: {list(rows[0])}')
        for r in rows:
            p = Path(r[pathcol])
            cls = r[labelcol] if labelcol else p.parent.name
            if cls not in by_class:
                continue
            # Manifests may hold absolute, repo-relative or bare names; resolve by name
            # under DATA_ROOT/test/<class>/ so it works regardless.
            cand = Path(DATA_ROOT) / 'test' / cls / p.name
            by_class[cls].append(cand if cand.exists() else p)
        n = sum(len(v) for v in by_class.values())
        print(f'clean manifest: {n} images  (expected 565)')
        if n != 565:
            print(f'  NOTE: manifest holds {n}, not the documented 565 — say which '
                  f'basis any number came from.')
        basis = f'de-duplicated clean test set from manifest (n={n})'
    else:
        print('WARNING: clean manifest NOT found. Falling back to the FULL 727-image '
              'test set, which is CONTAMINATED (22.3% have a pHash near-duplicate in '
              'train/val). Numbers from this basis are NOT comparable to the 565 set.')
        for cls in CLASSES:
            d = Path(DATA_ROOT) / 'test' / cls
            by_class[cls] = sorted(p for p in d.iterdir()
                                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        n = sum(len(v) for v in by_class.values())
        basis = f'FULL CONTAMINATED test set (n={n})'
    return by_class, basis


def sample_stratified(per_class: int, seed: int):
    """Deterministic stratified sample. Same construction as the local run, so the
    same (per_class, seed) reproduces the same images."""
    by_class, basis = load_clean_set()
    rows = []
    for cls in CLASSES:
        pool = sorted(by_class[cls], key=lambda p: p.name)   # stable before sampling
        rng = random.Random(f'{seed}:{cls}')
        take = min(per_class, len(pool))
        if take < per_class:
            print(f'  WARNING: {cls} has only {len(pool)} images, wanted {per_class}')
        rows.extend((p, cls) for p in rng.sample(pool, take))
    return rows, basis


FILES, BASIS = sample_stratified(PER_CLASS, SAMPLE_SEED)
print(f'basis: {BASIS}')
print(f'sampled {len(FILES)} images ({PER_CLASS}/class, seed {SAMPLE_SEED})')
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with (Path(OUT_DIR) / 'xai_aopc_sample.csv').open('w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['true_class', 'filename', 'path'])
    for p, c in FILES:
        w.writerow([c, p.name, str(p)])


# =====================================================================================
# 6. SCORE EVERY IMAGE
# =====================================================================================
OUT_CSV = Path(OUT_DIR) / 'xai_aopc_raw.csv'
fh = OUT_CSV.open('w', newline='', encoding='utf-8')
writer = csv.writer(fh)
writer.writerow(['path', 'true_class', 'pred_class', 'correct', 'pred_conf', 'method',
                 'seed', 'aopc', 'secs'] + [f'p{p}' for p in DELETION_PERCENTS])

verified = False
t_start = time.time()

for i, (path, true_cls) in enumerate(FILES, 1):
    tensor = preprocess_image(Image.open(path).convert('RGB'))
    base = ensemble_probs(tensor)
    # Guard the scale bug that produced p(class)=2.6 in the first run: a weighted
    # average of softmaxes must sum to 1 if the weights do.
    if i == 1 and not (0.99 <= float(base.sum()) <= 1.01):
        raise RuntimeError(
            f'ensemble probabilities sum to {base.sum():.4f}, expected 1.0 -- '
            f'ENSEMBLE_WEIGHTS are probably not normalised (got {ENSEMBLE_WEIGHTS})')
    class_idx = int(np.argmax(base))
    pred_cls = CLASSES[class_idx]
    pred_conf = float(base[class_idx])

    t0 = time.time()
    cam = ensemble_heatmap(tensor, class_idx)
    cam_secs = time.time() - t0

    jobs = [('gradcam', -1, cam, cam_secs)]
    for s in range(RANDOM_SEEDS):
        for name, grid in (('random7x7', CAM_GRID), ('randompixel', 224)):
            # Seed derived from the file name so a re-run reproduces the same maps.
            rng = np.random.default_rng(abs(hash((path.name, name, s))) % (2 ** 32))
            jobs.append((name, s, random_map(rng, grid), 0.0))

    if not verified:
        # Cross-check the batched scorer against the reference once, on the first real
        # image and the real Grad-CAM map, before trusting it for the whole run.
        ref = deletion_curve_reference(tensor, cam, class_idx)
        got = deletion_curve_batched(tensor, cam, class_idx)
        delta = max(abs(a - b) for a, b in zip(ref, got))
        if delta > 1e-4:
            raise RuntimeError(f'batched scorer disagrees with reference by {delta:.2e}')
        print(f'batched scorer verified against reference (max delta {delta:.2e})')
        verified = True

    for method, seed_i, hm, secs in jobs:
        curve = deletion_curve_batched(tensor, hm, class_idx)
        writer.writerow([str(path), true_cls, pred_cls, int(pred_cls == true_cls),
                         f'{pred_conf:.6f}', method, seed_i, f'{aopc(curve):.6f}',
                         f'{secs:.3f}'] + [f'{c:.6f}' for c in curve])
    fh.flush()

    if i % 5 == 0 or i == len(FILES):
        el = time.time() - t_start
        eta = el / i * (len(FILES) - i)
        print(f'[{i}/{len(FILES)}] {true_cls}/{path.name} pred={pred_cls} '
              f'({pred_conf:.3f}) cam={cam_secs:.1f}s  elapsed={el/60:.1f}m '
              f'eta={eta/60:.1f}m')

fh.close()
print(f'\ndone in {(time.time()-t_start)/60:.1f} min -> {OUT_CSV}')


# =====================================================================================
# 7. STATISTICS — paired Wilcoxon + sign test, implemented from scratch
# =====================================================================================
def _rankdata(a):
    order = np.argsort(a, kind='stable')
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon(d, exact_max=50):
    """Two-sided Wilcoxon signed-rank.

    Exact null for n <= exact_max when ranks are untied, via a dynamic program over the
    distribution of W+ (NOT 2**n enumeration -- that is 18 s at n=20 and hopeless above).
    With ties, or above exact_max, falls back to the normal approximation with tie and
    continuity correction. Verified to agree with scripts/xai_aopc_analyze.py on 520
    random trials across both paths.
    """
    d = np.asarray([x for x in d if x != 0.0], dtype=float)
    n = len(d)
    if n == 0:
        return float('nan'), 1.0, 0
    r = _rankdata(np.abs(d))
    w = min(float(r[d > 0].sum()), float(r[d < 0].sum()))
    has_ties = len(np.unique(np.abs(d))) != n

    if n <= exact_max and not has_ties:
        # ranks are exactly 1..n; count subsets summing to each value
        total_rank = n * (n + 1) // 2
        dp = np.zeros(total_rank + 1, dtype=float)
        dp[0] = 1.0
        for rank in range(1, n + 1):
            dp[rank:] += dp[:-rank].copy()
        dist = dp / dp.sum()
        tail = float(sum(dist[k] for k in range(total_rank + 1)
                         if min(k, total_rank - k) <= w + 1e-12))
        return w, min(1.0, tail), n

    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = sum(c ** 3 - c for c in counts)
    var = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    z = (w - mean + 0.5) / math.sqrt(var)
    return w, min(1.0, max(0.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2)))), n


def sign_test(d):
    pos = int(sum(1 for x in d if x > 0))
    n = int(sum(1 for x in d if x != 0))
    if n == 0:
        return 0, 0, 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    obs = pmf[pos]
    return pos, n, min(1.0, sum(p for p in pmf if p <= obs * (1 + 1e-12)))


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---- load back and aggregate: average each baseline's seeds BEFORE pairing ----
rows = list(csv.DictReader(OUT_CSV.open(encoding='utf-8-sig')))
per_image = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(list))
meta = {}
for r in rows:
    acc[r['path']][r['method']].append(float(r['aopc']))
    meta[r['path']] = (r['true_class'], r['pred_class'], int(r['correct']))
for path, methods in acc.items():
    for m, vals in methods.items():
        per_image[path][m] = float(np.mean(vals))      # seed-averaged

paths = [p for p in per_image if {'gradcam', 'random7x7', 'randompixel'} <= set(per_image[p])]
METHODS = ['gradcam', 'random7x7', 'randompixel']
vals = {m: np.array([per_image[p][m] for p in paths]) for m in METHODS}

classes_present = sorted({meta[p][0] for p in paths})
lines = []
say = lines.append
say('# Grad-CAM faithfulness: deletion/AOPC vs random baselines (Kaggle T4)')
say('')
say(f'n = {len(paths)} images. Basis: {BASIS}.')
say(f'Classes present: {", ".join(classes_present)} ({len(classes_present)}/5).')
say('')
if len(classes_present) < 5:
    say(f'**PARTIAL RUN — only {len(classes_present)} of 5 classes present. NOT A '
        'RESULT: the sample is ordered by class, so an incomplete run is class-biased, '
        'not a random subset.**')
    say('')
say('## Overall AOPC (higher = heatmap found pixels the prediction depends on)')
say('')
say('| method | mean AOPC | 95% CI (bootstrap) | median |')
say('|---|---|---|---|')
for m in METHODS:
    lo, hi = boot_ci(vals[m])
    say(f'| {m} | {vals[m].mean():.4f} | [{lo:.4f}, {hi:.4f}] | '
        f'{float(np.median(vals[m])):.4f} |')
say('')
say('## Paired comparisons (Grad-CAM minus baseline, same images)')
say('')
say('| comparison | mean diff | 95% CI | Wilcoxon W | p | sign test | p |')
say('|---|---|---|---|---|---|---|')
for b in METHODS[1:]:
    d = vals['gradcam'] - vals[b]
    lo, hi = boot_ci(d)
    w, p, nz = wilcoxon(d)
    pos, nn, sp = sign_test(d)
    say(f'| gradcam - {b} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {w:.1f} (n={nz}) '
        f'| {p:.4f} | {pos}/{nn} positive | {sp:.4f} |')
say('')
say('**`random7x7` is the fair comparator; `randompixel` is a sanity check, not a '
    'rival.** Both delete the same NUMBER of pixels but not the same KIND of region. '
    '`random7x7` draws noise on Grad-CAM\'s native 7x7 grid and upsamples it, so its '
    'deletions are contiguous blobs matched in spatial structure to a Grad-CAM map — '
    'the only difference is whether the blobs land in informative places. '
    '`randompixel` scatters i.i.d. noise at full resolution, shredding local texture '
    'everywhere and pushing the image far off the training distribution, which '
    'collapses p(class) no matter what it covers. That is why it posts the highest '
    'AOPC while explaining nothing. Read `gradcam - random7x7` as the result; treat '
    '`gradcam - randompixel` as confirmation that AOPC is sensitive to '
    'out-of-distribution damage and so must only be compared against a '
    'structure-matched baseline.')
say('')
d = vals['gradcam'] - vals['random7x7']
wins = int((d > 0).sum())
dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float('nan')
say('## Effect size vs random7x7')
say('')
say(f'- Grad-CAM scored higher on **{wins}/{len(d)}** images ({wins/len(d)*100:.1f}%).')
say(f'- Mean paired difference {d.mean():+.4f}, SD {d.std(ddof=1):.4f}, '
    f"Cohen's dz = {dz:.3f}.")
say('')
say('## Per-class AOPC (each row individually underpowered; CIs are wide)')
say('')
say('| true class | n | gradcam | random7x7 | randompixel | gradcam-random7x7 | p |')
say('|---|---|---|---|---|---|---|')
for cls in CLASSES:
    sub = [p for p in paths if meta[p][0] == cls]
    if not sub:
        continue
    g = np.array([per_image[p]['gradcam'] for p in sub])
    r7 = np.array([per_image[p]['random7x7'] for p in sub])
    rp = np.array([per_image[p]['randompixel'] for p in sub])
    _, pv, _ = wilcoxon(g - r7)
    say(f'| {cls} | {len(sub)} | {g.mean():.4f} | {r7.mean():.4f} | {rp.mean():.4f} | '
        f'{(g-r7).mean():+.4f} | {pv:.4f} |')
say('')
say('## Split by whether the ensemble was correct')
say('')
say('| subset | n | gradcam | random7x7 | gradcam-random7x7 | p |')
say('|---|---|---|---|---|---|')
for label, want in (('correct', 1), ('wrong', 0)):
    sub = [p for p in paths if meta[p][2] == want]
    if not sub:
        continue
    g = np.array([per_image[p]['gradcam'] for p in sub])
    r7 = np.array([per_image[p]['random7x7'] for p in sub])
    _, pv, _ = wilcoxon(g - r7)
    say(f'| {label} | {len(sub)} | {g.mean():.4f} | {r7.mean():.4f} | '
        f'{(g-r7).mean():+.4f} | {pv:.4f} |')
say('')
say('## Mean deletion curve: p(predicted class) as the top-k pixels are blanked')
say('')
say('| deleted % | ' + ' | '.join(METHODS) + ' |')
say('|---|' + '---|' * len(METHODS))
curves = {m: defaultdict(list) for m in METHODS}
for r in rows:
    for pct in DELETION_PERCENTS:
        curves[r['method']][pct].append(float(r[f'p{pct}']))
for pct in DELETION_PERCENTS:
    say(f'| {pct}% | ' + ' | '.join(f'{np.mean(curves[m][pct]):.4f}' for m in METHODS)
        + ' |')

report = '\n'.join(lines)
(Path(OUT_DIR) / 'xai_aopc_analysis.md').write_text(report, encoding='utf-8')
print(report)
print(f'\nwrote {OUT_DIR}/xai_aopc_analysis.md')
print(f'wrote {OUT_CSV}')
print(f'wrote {OUT_DIR}/xai_aopc_sample.csv')
print('\nDownload all three from /kaggle/working and drop the CSV into .nb1out/ locally.')
