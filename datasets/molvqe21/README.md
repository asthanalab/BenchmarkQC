# MolVQE-21 corrected reference systems

This folder contains the corrected MolVQE-21 reference Hamiltonians imported
from Mushir’s `asthanaa/benchmarkkrylov` checkout at commit
`2f1faef5589dd276393be4cb93860524f4ff790a`. The source commit consolidates 27
benchmark-ready Hamiltonian caches and applies explicit canonical-MO active
orbital corrections to cyclobutadiene and the two m-benzyne cases.

Each benchmark-ready case uses the same per-system layout as BenchmarkQC:

- `systems/<case_id>/hamiltonian.npz`: the standard Benchmark-QC `labels`,
  `Hs`, and `casci_energies` archive, with one point labeled `0.0`.
- `systems/<case_id>/inputs/source_integrals.npz`: a normalized, pickle-free
  numeric integral archive that can regenerate the Jordan–Wigner terms.
- `systems/<case_id>/metadata.json`: the shared per-system metadata and
  reference-result record.

`source/manifest.csv` is the common family manifest. The corrected-cache
manifest, filtered source manifest, active-orbital overrides, and builder are
kept alongside it under `source/`. The package manifest is deliberately
filtered to the 27 corrected-cache cases; the two incomplete source records
are excluded from both the folder and the benchmark catalog. The existing
corrected Hamiltonian and integral archives were moved and indexed; no new
electronic-structure calculation was needed for this layout normalization.

The family-level reference indexes use the same names as BenchmarkQC:
`reference/reference_results.json` and `reference/inventory.json`.

The two scalar reference fields not published by the MolVQE-21 source are
represented as `null` with status `not-provided-by-source`; this preserves the
source boundary instead of inventing CISD or CCSD values.

The package loader is available as:

```python
from benchmark_qc.molvqe21 import list_cases, load_hamiltonian, load_source_integrals

case = list_cases()[0]
hamiltonian = load_hamiltonian(case.case_id)
integrals = load_source_integrals(case.case_id)
```

To regenerate this folder from a fresh `benchmarkkrylov` checkout, run
`source/build_dataset.py` with `--source-root` pointing to that checkout.
