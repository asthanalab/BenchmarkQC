# Package API

## Installation (PyPI)

To install the package from PyPI:

```sh
pip install benchmark-qc
```

The importable package lives in `src/benchmark_qc/` and is installed as
`benchmark_qc`.

## `benchmark_qc.hamiltonian_test`

Utilities used by all `test_*.py` scripts.

- `load_hamiltonian_npz(npz_path: str) -> NPZData`
  - Loads `labels`, `Hs`, `casci_energies` from a saved PES `.npz`.

- `pick_point_index(labels, *, index: int | None, bond: float | None) -> int`
  - Selects a geometry point by explicit index or nearest bond-length label.

- `ground_energy_from_terms(terms, *, nelec: int | None = None, spin: int | None = None) -> (float, str)`
  - Computes the Hamiltonian ground energy by diagonalization.
  - If `nelec` and `spin` are provided, restricts diagonalization to that sector.

- `sector_matrix_from_terms(terms, *, nelec: int, spin: int)`
  - Projects Pauli terms directly into the requested fixed electron/spin sector.
  - Avoids constructing the full Fock-space matrix for the 16-qubit N2 data.

## `benchmark_qc.integral_dataset`

- `load_spatial_integral_archive(path) -> SpatialIntegralArchive`
  - Loads and validates a pickle-free active-space integral/orbital archive.
- `jordan_wigner_terms_from_integrals(archive, *, cutoff=1e-14) -> np.ndarray`
  - Converts PySCF chemist-ordered spatial integrals to PennyLane
    Jordan-Wigner terms using the repository's interleaved spin order.
- `write_legacy_hamiltonian_npz(...)`
  - Writes the backward-compatible three-array dataset contract atomically.
- `max_pauli_coefficient_difference(left, right) -> float`
  - Compares two Hamiltonians coefficient-by-coefficient after canonicalizing
    their Pauli words.

## `benchmark_qc.n2`

- `H_gen(...) -> (H, cas_energy)`
  - Generates a PennyLane qubit Hamiltonian and a CASCI reference energy.

## `benchmark_qc.fes`

- `H_gen(...) -> (H, cas_energy, operator_pool)`
  - Generates a PennyLane qubit Hamiltonian, CASCI reference energy, and the excitation operator pool.

## `benchmark_qc.fe2s2`

- `load_hamiltonian(variant="historical_cas6e6o") -> NPZData`
  - Loads an accepted historical or constructed Fe2S2 PennyLane Hamiltonian archive.
- `load_source_integrals(variant="historical_cas6e6o") -> SpatialIntegralArchive`
  - Loads the corresponding checksum-pinned, pickle-free integral archive.
- `accepted_variant_available(variant) -> bool`
  - Verifies the promoted report and both artifact checksums.  For the
    constructed CAS(8e,8o) control it also verifies the independent
    cross-version report, certificate, and their bound checksums.

Accepted variants are `historical_cas4e4o`, `historical_cas6e6o`, and
`historical_cas8e6o`, plus the distinct `nested_cas8e8o` present-work control.
They preserve the qualified nonstationary default-RHF frame and are not
natural-orbital or Active Space Finder datasets.  `nested_cas8e8o` activates
parent MOs 11--18, strictly contains the historical CAS(6e,6o) span, and is
not a historical paper active space.

## `benchmark_qc.molvqe21`

- `list_cases() -> tuple[MolVQE21Case, ...]`
  - Lists the 27 corrected-cache MolVQE-21 reference cases.
- `get_case(case_id) -> MolVQE21Case`
  - Returns stable metadata for one case.
- `load_hamiltonian(case_id) -> NPZData`
  - Loads the standard one-point Benchmark-QC Hamiltonian archive.
- `load_source_integrals(case_id) -> SpatialIntegralArchive`
  - Loads the corresponding normalized, pickle-free numeric integral archive.

The package intentionally excludes the two MolVQE source records without
corrected benchmark caches: `Ferrocene_ceo` and `feo4_2minus_ceo`.

## `benchmark_qc.u2`

- `build_u2_reference(basis_input, ncas, nelecas) -> (mol_ref, mo_ref)`
- `H_gen(..., mol_ref, mo_ref, ...) -> (H, casci_energy)`

## Notebook compatibility wrappers

These files exist so your existing notebooks keep working without changes:

- `datasets/benchmarkQC/N2/ham_pyscf.py`
- `datasets/benchmarkQC/FeS/FeS_pyscf.py`
- `datasets/benchmarkQC/U2/U2_ham1.py`
