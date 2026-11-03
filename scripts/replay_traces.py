"""A2 replay harness: run real images through the full screening pipeline.

Pairs held-out test images with synthetic PatientMetadata and calls
run_screening end-to-end, writing structured JSONL for error analysis.

    # Dry run (no API cost, FakeClient):
    python scripts/replay_traces.py --config configs/eval/replay_v1.yaml --dry-run

    # Real API:
    SKINSURE_ALLOW_SPEND=1 python scripts/replay_traces.py \
        --config configs/eval/replay_v1.yaml --max-usd 0.10

    # Resume a partial run:
    python scripts/replay_traces.py --config configs/eval/replay_v1.yaml --dry-run --resume
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

from src.agents.orchestrator import run_screening  # noqa: E402
from src.eval.synthetic_patient import generate_patients_iter  # noqa: E402
from src.explainability.explainer import GradCamExplainer  # noqa: E402
from src.models.ensemble import SkinEnsemble  # noqa: E402
from src.utils.config import get_checkpoint_paths, get_model_config  # noqa: E402

if __package__:
    from ._guard import SpendGuard, SpendLimitReached, add_spend_args
else:
    from _guard import SpendGuard, SpendLimitReached, add_spend_args

CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]
MANIFEST = Path(r"C:\Users\Cipher\Downloads\test_clean_manifest.csv")
TEST_ROOT = ROOT / "data" / "raw" / "IPD_Datasets_Final_Split" / "test"


# ---------------------------------------------------------------------------
# File list (reuses the hair_ab pattern)
# ---------------------------------------------------------------------------

def build_file_list(
    per_class: int, seed: int,
) -> tuple[list[tuple[Path, str]], str]:
    on_disk: dict[str, dict[str, Path]] = {}
    for cls in CLASSES:
        on_disk[cls] = {p.name: p for p in (TEST_ROOT / cls).iterdir() if p.is_file()}

    rows: list[tuple[Path, str]] = []
    if MANIFEST.exists():
        missing = 0
        with MANIFEST.open(encoding="utf-8-sig") as fh:
            for rec in csv.DictReader(fh):
                man = PurePosixPath(rec["path"])
                cls, name = man.parent.name, man.name
                if cls not in on_disk:
                    continue
                hit = on_disk[cls].get(name)
                if hit is None:
                    missing += 1
                    continue
                rows.append((hit, cls))
        basis = f"clean manifest (n={len(rows)}, unmatched={missing})"
    else:
        for cls in CLASSES:
            rows.extend((p, cls) for p in sorted(on_disk[cls].values()))
        basis = f"CONTAMINATED full test dir (n={len(rows)})"

    rng = random.Random(seed)
    sampled: list[tuple[Path, str]] = []
    for cls in CLASSES:
        pool = sorted([r for r in rows if r[1] == cls], key=lambda r: r[0].name)
        rng.shuffle(pool)
        sampled.extend(pool[:per_class])
    return sampled, f"{basis} -> stratified {per_class}/class, seed {seed}, n={len(sampled)}"


# ---------------------------------------------------------------------------
# Metered client wrapper (from summarize_one.py pattern)
# ---------------------------------------------------------------------------

class _MeteredClient:
    def __init__(self, inner: anthropic.AsyncAnthropic, guard: SpendGuard):
        self._inner = inner
        self._guard = guard
        self.messages = type("_M", (), {"create": self._create})()

    async def _create(self, **kwargs):
        self._guard.check()
        response = await self._inner.messages.create(**kwargs)
        cost = self._guard.record(response.usage)
        return response


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def _result_row(
    image_path: str,
    true_class: str,
    result,
    patient,
    elapsed: float,
    dry_run: bool,
    error: str | None = None,
) -> dict:
    row: dict = {
        "image_path": image_path,
        "true_class": true_class,
        "elapsed_s": round(elapsed, 3),
        "dry_run": dry_run,
        "error": error,
    }

    if error is not None:
        return row

    r = result
    p = r.predictions
    s = r.summary
    x = r.xai

    row["status"] = r.status.value if r.status else None
    row["primary_label"] = p.primary_label.value if p else None
    row["primary_confidence"] = round(p.primary_confidence, 6) if p else None
    row["secondary_label"] = p.secondary_label.value if p else None
    row["secondary_confidence"] = round(p.secondary_confidence, 6) if p else None
    row["agreement_score"] = round(p.agreement_score, 6) if p else None
    row["confidence_tier"] = p.confidence_tier() if p else None

    row["severity"] = s.severity.value if s else None
    row["confidence_level"] = s.confidence_level.value if s else None
    row["recapture_needed"] = s.recapture_needed if s else None
    row["immediate_referral"] = s.immediate_referral if s else None
    row["summary_text"] = s.summary if s else None
    row["recommended_action"] = s.recommended_action if s else None
    row["confidence_caveats"] = s.confidence_caveats if s else None

    row["xai_focus"] = x.focus_description.value if x else None
    row["xai_boundary"] = x.boundary_alignment.value if x else None
    row["xai_artifact_flag"] = x.artifact_flag if x else None

    row["triage_reasons"] = r.triage_reasons
    row["escalated"] = r.escalated
    row["treatment_advice_allowed"] = r.treatment_advice_allowed
    row["warnings"] = r.warnings
    row["messages"] = r.messages

    row["patient_fitzpatrick"] = patient.fitzpatrick_type.value
    row["patient_age"] = patient.patient_age
    row["patient_is_spreading"] = patient.is_spreading_rapidly

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2 replay: images through the full pipeline.")
    parser.add_argument("--config", required=True, help="YAML config (e.g. configs/eval/replay_v1.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Use FakeClient instead of the real API")
    parser.add_argument("--resume", action="store_true", help="Skip images already in the output JSONL")
    add_spend_args(parser)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_done(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("image_path"):
                    done.add(row["image_path"])
            except json.JSONDecodeError:
                continue
    return done


async def main():
    args = parse_args()
    cfg = load_config(args.config)
    sample_cfg = cfg["sample"]
    output_cfg = cfg["output"]

    results_path = ROOT / output_cfg["results"]
    manifest_path = ROOT / output_cfg["sample_manifest"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    per_class = sample_cfg["per_class"]
    seed = sample_cfg["seed"]
    spreading_rate = sample_cfg.get("spreading_rate", 0.15)

    files, basis = build_file_list(per_class, seed)
    total = len(files)
    print(f"Sample: {basis}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_class"])
        for p, cls in files:
            w.writerow([p.relative_to(TEST_ROOT).as_posix(), cls])
    print(f"Manifest saved: {manifest_path}")

    done: set[str] = set()
    if args.resume:
        done = _load_done(results_path)
        if done:
            print(f"Resuming: {len(done)} images already processed, skipping them")

    print("Loading ensemble and explainer...")
    w_cfg = get_model_config().weights
    ensemble = SkinEnsemble(
        get_checkpoint_paths(),
        weights=(w_cfg["efficientnet"], w_cfg["convnext"], w_cfg["swin"]),
    )
    explainer = GradCamExplainer(ensemble)

    guard: SpendGuard | None = None
    if args.dry_run:
        sys.path.insert(0, str(ROOT / "tests"))
        from helpers import FakeClient, llm_json
        client = FakeClient(llm_json())
        print("DRY RUN: using FakeClient (no API cost)")
    else:
        load_dotenv(override=True)
        guard = SpendGuard.from_args(args)
        real_client = anthropic.AsyncAnthropic()
        client = _MeteredClient(real_client, guard)
        print(f"LIVE RUN: max ${args.max_usd:.4f}")

    patient_iter = generate_patients_iter(seed=seed, spreading_rate=spreading_rate)

    completed = 0
    errors = 0
    skipped = 0
    mode = "a" if args.resume else "w"

    with results_path.open(mode, encoding="utf-8") as out:
        for idx, ((img_path, true_class), patient) in enumerate(zip(files, patient_iter)):
            rel = img_path.relative_to(TEST_ROOT).as_posix()

            if rel in done:
                skipped += 1
                continue

            t0 = time.perf_counter()
            try:
                pil = Image.open(img_path).convert("RGB")
                result = await run_screening(
                    pil, patient,
                    ensemble=ensemble,
                    explainer=explainer,
                    client=client,
                )
                elapsed = time.perf_counter() - t0

                row = _result_row(rel, true_class, result, patient, elapsed, args.dry_run)

                label = row.get("primary_label", "?")
                conf = row.get("primary_confidence")
                sev = row.get("severity", "?")
                conf_str = f"{conf:.2f}" if conf is not None else "?"
                print(f"[{completed + 1}/{total - skipped}] {rel}  -> {label} ({conf_str})  {sev}  {elapsed:.1f}s")

            except SpendLimitReached as e:
                print(f"\n{e}")
                break
            except Exception as e:
                elapsed = time.perf_counter() - t0
                row = _result_row(rel, true_class, None, patient, elapsed, args.dry_run, error=str(e))
                errors += 1
                print(f"[{completed + 1}/{total - skipped}] {rel}  ERROR: {e}")
                traceback.print_exc()

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            completed += 1

    print(f"\nDone: {completed} processed, {errors} errors, {skipped} skipped (resume)")
    print(f"Results: {results_path}")
    if guard:
        print(f"Spend: {guard.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
