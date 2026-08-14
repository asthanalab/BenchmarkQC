# Repository layout

Benchmark-QC separates reusable Python code, checked-in reference data, and
local application work so a dataset commit remains reviewable.

```text
BenchmarkQC/
├── datasets/
│   ├── catalog.json          # authoritative inventory and checksums
│   ├── benchmarkQC/          # all BenchmarkQC and SI Table I families
│   └── molvqe21/             # 27 corrected MolVQE-21 cases
├── src/benchmark_qc/         # installable Python package
├── tests/                    # repository-wide regression tests
├── docs/                     # user, format, API, and release documentation
├── applications/             # ignored local calculation outputs
├── pyproject.toml
└── .github/workflows/        # CI
```

## What belongs where

Dataset archives, source integrals, metadata, provenance, and reproducibility
builders belong beside the dataset family under `datasets/`. Shared loading,
conversion, and validation code belongs under `src/benchmark_qc/`. Cross-family
tests belong under `tests/`; a small system-specific validation script may stay
beside the system it documents.

Actual algorithm runs, hardware submissions, optimizer traces, plots, logs,
and derived tables belong under `applications/`. Those contents are ignored by
Git so an application experiment cannot silently become part of the reference
dataset. Keep reusable application code outside those ignored output folders.

The catalog is always interpreted relative to the repository root. This makes
clones, CI runners, and different user workspaces portable.
