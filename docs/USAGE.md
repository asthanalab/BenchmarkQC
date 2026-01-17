# Usage

This repo contains benchmark qubit Hamiltonians for:
- N2 (folder: `N2/`)
- FeS (folder: `FeS/`)
- U2 (folder: `U2/`)

Each folder contains:
- an input notebook (`*-in*.ipynb`) that generated a PES
- a saved `.npz` file containing Hamiltonians + reference energies
- a small test script (`test_*.py`) that validates the saved Hamiltonian

## Quickstart (recommended)

Create and use a dedicated conda environment:

- `conda create -n bench python=3.11`
- `conda install -n bench -c conda-forge numpy scipy pyscf sympy basis_set_exchange`
- `conda run -n bench python -m pip install pennylane`
- `conda run -n bench python -m pip install -e .`

## Run Hamiltonian sanity checks

From the repo root:

- `conda run -n bench python N2/test_n2_hamiltonian.py --index 0`
- `conda run -n bench python FeS/test_fes_hamiltonian.py --index 0`
- `conda run -n bench python U2/test_u2_hamiltonian.py --index 0`

You can also choose a point by bond length (nearest stored label):

- `conda run -n bench python N2/test_n2_hamiltonian.py --bond 1.4`
- `conda run -n bench python FeS/test_fes_hamiltonian.py --bond 2.4`
- `conda run -n bench python U2/test_u2_hamiltonian.py --bond 2.48`

## Stored geometry points (bond lengths)

In each saved `.npz`, the `labels` array stores the geometry “point label” used at
generation time. For these diatomics, it is the bond length in **Angstrom** (PySCF
default unit, since the generators do not override `mol.unit`).

- **N2** (`N2/N2_PES_H1.npz`, 11 points):
	- $R$ (Å) = 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
- **FeS** (`FeS/FeS_PES_H.npz`, 14 points):
	- $R$ (Å) = 1.826, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 3.7, 4.0, 4.2, 4.5, 4.8
- **U2** (`U2/U2_PES_H.npz`, 2 points):
	- $R$ (Å) = 2.4, 2.48

## Important: FeS is open-shell/high-spin

FeS is an open-shell system. The saved reference energy is for a *specific* electron/spin sector.

The FeS test script compares energies in that same sector:
- `--nelec` = active-space electrons (default `6`, matches the FeS notebook)
- `--spin`  = PySCF convention `spin = 2S` (default `4`, matches the FeS notebook)

If you change the physical state you want to target (different `spin`), the stored `casci_energies`
will only match if the `.npz` was generated with that same choice.

## Generating Hamiltonians (not required for tests)

The Hamiltonian-generation functions are exposed for notebook compatibility:

- `N2/ham_pyscf.py` exports `H_gen`
- `FeS/FeS_pyscf.py` exports `H_gen`
- `U2/U2_ham1.py` exports `build_u2_reference` and `H_gen`

The actual implementations live in the package:
- `benchmark_qc.n2.H_gen`
- `benchmark_qc.fes.H_gen`
- `benchmark_qc.u2.build_u2_reference`, `benchmark_qc.u2.H_gen`

Some generation code depends on an external active-space finder (`asf`). The test scripts do not.
