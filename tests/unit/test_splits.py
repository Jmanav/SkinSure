"""Tests for scripts/make_splits.py and scripts/make_federated_shards.py."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EXPECTED_CLASSES = ["Basal_Cell_Carcinoma", "Eczema", "Keratosis", "Melanoma", "Tinea"]


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a flat dataset with 50 images per class."""
    for cls in EXPECTED_CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(50):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(cls_dir / f"{i}.jpg")
    return tmp_path


@pytest.fixture
def presplit_dataset(tmp_path):
    """Create a pre-split dataset (train/val/test layout)."""
    for split in ["train", "val", "test"]:
        for cls in EXPECTED_CLASSES:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            n = {"train": 30, "val": 10, "test": 10}[split]
            for i in range(n):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(d / f"{i}.jpg")
    return tmp_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestMakeSplits:
    def test_flat_dataset(self, fake_dataset, tmp_path):
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "scripts/make_splits.py",
             "--data-dir", str(fake_dataset),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        assert result.returncode == 0, result.stderr

        pool = read_csv(out_dir / "trainable_pool.csv")
        test = read_csv(out_dir / "global_test.csv")

        assert len(pool) + len(test) == 250
        assert len(test) >= 25

        test_labels = {r["label"] for r in test}
        pool_labels = {r["label"] for r in pool}
        assert test_labels == set(EXPECTED_CLASSES)
        assert pool_labels == set(EXPECTED_CLASSES)

    def test_presplit_dataset(self, presplit_dataset, tmp_path):
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "scripts/make_splits.py",
             "--data-dir", str(presplit_dataset),
             "--presplit",
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        assert result.returncode == 0, result.stderr

        pool = read_csv(out_dir / "trainable_pool.csv")
        test = read_csv(out_dir / "global_test.csv")
        assert len(pool) + len(test) == 250

    def test_no_overlap(self, fake_dataset, tmp_path):
        out_dir = tmp_path / "out"
        subprocess.run(
            [sys.executable, "scripts/make_splits.py",
             "--data-dir", str(fake_dataset),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        pool_paths = {r["path"] for r in read_csv(out_dir / "trainable_pool.csv")}
        test_paths = {r["path"] for r in read_csv(out_dir / "global_test.csv")}
        assert pool_paths.isdisjoint(test_paths)


class TestMakeFederatedShards:
    @pytest.fixture
    def pool_csv(self, fake_dataset, tmp_path):
        out_dir = tmp_path / "splits"
        subprocess.run(
            [sys.executable, "scripts/make_splits.py",
             "--data-dir", str(fake_dataset),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        return out_dir / "trainable_pool.csv"

    def test_iid_shards(self, pool_csv, tmp_path):
        out_dir = tmp_path / "shards"
        result = subprocess.run(
            [sys.executable, "scripts/make_federated_shards.py",
             "--strategy", "iid",
             "--pool", str(pool_csv),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        assert result.returncode == 0, result.stderr

        total = 0
        for name in ["hospital_a", "hospital_b", "hospital_c"]:
            rows = read_csv(out_dir / f"{name}.csv")
            assert len(rows) > 0
            total += len(rows)

        pool_rows = read_csv(pool_csv)
        assert total == len(pool_rows)

    def test_noniid_shards(self, pool_csv, tmp_path):
        out_dir = tmp_path / "shards"
        result = subprocess.run(
            [sys.executable, "scripts/make_federated_shards.py",
             "--strategy", "noniid",
             "--pool", str(pool_csv),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        assert result.returncode == 0, result.stderr

        total = 0
        for name in ["hospital_a", "hospital_b", "hospital_c"]:
            rows = read_csv(out_dir / f"{name}.csv")
            assert len(rows) > 0
            total += len(rows)

        pool_rows = read_csv(pool_csv)
        assert total == len(pool_rows)

    def test_no_overlap_between_shards(self, pool_csv, tmp_path):
        out_dir = tmp_path / "shards"
        subprocess.run(
            [sys.executable, "scripts/make_federated_shards.py",
             "--strategy", "noniid",
             "--pool", str(pool_csv),
             "--output-dir", str(out_dir),
             "--seed", "42"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        all_paths: list[str] = []
        for name in ["hospital_a", "hospital_b", "hospital_c"]:
            rows = read_csv(out_dir / f"{name}.csv")
            all_paths.extend(r["path"] for r in rows)
        assert len(all_paths) == len(set(all_paths))
