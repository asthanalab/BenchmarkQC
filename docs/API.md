# Package API

The importable package lives in `benchmark_qc/`.

## `benchmark_qc.hamiltonian_test`

Utilities used by all `test_*.py` scripts.

- `load_hamiltonian_npz(npz_path: str) -> NPZData`
  - Loads `labels`, `Hs`, `casci_energies` from a saved PES `.npz`.

- `pick_point_index(labels, *, index: int | None, bond: float | None) -> int`
  - Selects a geometry point by explicit index or nearest bond-length label.

- `ground_energy_from_terms(terms, *, nelec: int | None = None, spin: int | None = None) -> (float, str)`
  - Computes the Hamiltonian ground energy by diagonalization.
  - If `nelec` and `spin` are provided, restricts diagonalization to that sector.

## `benchmark_qc.n2`

- `H_gen(...) -> (H, cas_energy)`
  - Generates a PennyLane qubit Hamiltonian and a CASCI reference energy.

## `benchmark_qc.fes`

- `H_gen(...) -> (H, cas_energy, operator_pool)`
  - Generates a PennyLane qubit Hamiltonian, CASCI reference energy, and the excitation operator pool.

## `benchmark_qc.u2`

- `build_u2_reference(basis_input, ncas, nelecas) -> (mol_ref, mo_ref)`
- `H_gen(..., mol_ref, mo_ref, ...) -> (H, casci_energy)`

## Notebook compatibility wrappers

These files exist so your existing notebooks keep working without changes:

- `N2/ham_pyscf.py`
- `FeS/FeS_pyscf.py`
- `U2/U2_ham1.py`
