# Benchmark-QC

Minimal utilities for generating and validating qubit Hamiltonians saved in the `N2/`, `FeS/`, and `U2/` folders.

## Package

This repo now contains a small Python package: `benchmark_qc/`.

Optional install (recommended for notebooks/usage from subfolders):

- `pip install -e .`

## Hamiltonian sanity checks

From the repo root:

- `python N2/test_n2_hamiltonian.py --index 0`
- `python FeS/test_fes_hamiltonian.py --index 0`
- `python U2/test_u2_hamiltonian.py --index 0`
