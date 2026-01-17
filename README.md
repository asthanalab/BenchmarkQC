# Benchmark-QC

Benchmark qubit Hamiltonians for N2, FeS, and U2.

This repo stores pre-generated Hamiltonians in `.npz` files (one per system) and provides:
- A small Python package (`benchmark_qc/`) with shared utilities
- Backwards-compatible notebook wrappers in each system folder
- Simple test scripts that validate the saved Hamiltonians

## Documentation

Start here: [docs/README.md](docs/README.md)

## Install (recommended)

Editable install so notebooks/scripts can import the package from any folder:

- `pip install -e .`

If you prefer conda, see [docs/USAGE.md](docs/USAGE.md) for the `bench` environment setup.

## Run Hamiltonian sanity checks

From the repo root:

- `python N2/test_n2_hamiltonian.py --index 0`
- `python FeS/test_fes_hamiltonian.py --index 0`
- `python U2/test_u2_hamiltonian.py --index 0`

## Notes

- FeS is open-shell/high-spin. The FeS test restricts diagonalization to the same `(N_alpha, N_beta)` sector as the stored CASCI reference (see [docs/USAGE.md](docs/USAGE.md)).
- The saved `.npz` files include a `labels` array (bond lengths). The exact per-system grids are listed in [docs/USAGE.md](docs/USAGE.md) under “Stored geometry points (bond lengths)”.
