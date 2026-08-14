# MolVQE-21 corrected reference systems

This folder contains the corrected MolVQE-21 reference Hamiltonians imported
from Mushir’s `asthanaa/benchmarkkrylov` checkout at commit
`2f1faef5589dd276393be4cb93860524f4ff790a`. The source commit consolidates 27
benchmark-ready Hamiltonian caches and applies explicit canonical-MO active
orbital corrections to cyclobutadiene and the two m-benzyne cases.

Each benchmark-ready case has two files:

- `hamiltonians/<case_id>.npz`: the standard Benchmark-QC `labels`, `Hs`, and
  `casci_energies` archive, with one point labeled `0.0`.
- `integrals/<case_id>.npz`: a normalized, pickle-free numeric integral archive
  that can regenerate the Jordan–Wigner terms.

`manifest.csv` records checksums, system metadata, source cache provenance, and
the active-orbital selection status. The package manifest is deliberately
filtered to the 27 corrected-cache cases; the two incomplete source records
are excluded from both the folder and the benchmark catalog.

The package loader is available as:

```python
from benchmark_qc.molvqe21 import list_cases, load_hamiltonian, load_source_integrals

case = list_cases()[0]
hamiltonian = load_hamiltonian(case.case_id)
integrals = load_source_integrals(case.case_id)
```

To regenerate this folder from a fresh `benchmarkkrylov` checkout, run
`build_dataset.py` with `--source-root` pointing to that checkout.
