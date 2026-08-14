"""Repository paths shared by the Benchmark-QC loaders."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPOSITORY_ROOT / "datasets"
BENCHMARKQC_ROOT = DATASETS_ROOT / "benchmarkQC"
MOLVQE21_ROOT = DATASETS_ROOT / "molvqe21"
CATALOG_PATH = DATASETS_ROOT / "catalog.json"
