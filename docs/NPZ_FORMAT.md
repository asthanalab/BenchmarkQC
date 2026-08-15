# Saved NPZ Format

## Installation (PyPI)

To install the package from PyPI:

```sh
pip install benchmark-qc
```

The Hamiltonian `.npz` files (e.g.
`datasets/benchmarkQC/C2/cas8e8o_augccpvtz/casci_natural_orbitals/C2_PES_H.npz`,
`datasets/benchmarkQC/N2/N2_PES_H1.npz`, `datasets/benchmarkQC/FeS/FeS_PES_H.npz`,
`datasets/benchmarkQC/Fe2S2/cas6e6o_chan30e20o/accepted/historical_cas6e6o/Fe2S2_H.npz`,
and `datasets/benchmarkQC/U2/U2_PES_H.npz`) share the same format.

They contain three arrays:

## `labels`

- dtype: `object`
- shape: `(n_points,)`
- meaning: the scanned geometry label for each point.

For the diatomic scans, this is the **bond length in Angstrom** (a float),
following PySCF's default unit where the generators do not override `mol.unit`.
For the historical Fe2S2 archives, the sole label `0.0` is a point identifier,
not a bond length, because the parent FCIDUMP does not encode coordinates.

For the exact labels used by each system (C2, C2H4, CH2, Fe2S2, FeH, FeS, N2,
O2, and U2), see
`docs/USAGE.md` →
“Stored geometry point labels.”

## `Hs`

- dtype: `object`
- shape: `(n_points,)`

Each entry `Hs[i]` is itself an *object array of PennyLane Pauli terms*.
Conceptually this is a sum of terms like:

- `(coeff) * I(0)`
- `(coeff) * (Z(0) @ Z(2))`

These are PennyLane operator objects (commonly `SProd` etc).

## `casci_energies`

- dtype: `float64`
- shape: `(n_points,)`

Reference energies (Hartree) computed alongside the Hamiltonian generation.

## Notes on diagonalization

- For small Hamiltonians (e.g. N2: 8 qubits → 256×256), tests can diagonalize densely.
- For larger ones (e.g. datasets/benchmarkQC/FeS/U2: 12 qubits → 4096×4096), tests use sparse eigen-solvers.
- For open-shell/high-spin systems (FeS), you must compare in the same `(N_alpha, N_beta)` sector.
  The FeS test does this using `--nelec` and `--spin`.
- Closed-shell active-space references must also be checked in the stated
  electron/spin sector. This prevents an unintended particle-number sector from
  being mistaken for the CASCI target.

## Trust and portability

`Hs` is an object array containing pickled PennyLane operators, so
`allow_pickle=True` is required. Python pickle can execute code while loading;
load these files only from a trusted source and verify their SHA-256 values
against `datasets/catalog.json`.

The current numeric system archives include pickle-free spatial-integral input
alongside the historical Hamiltonian representation. These store the available
geometry information, spatial one- and two-electron integrals, core constant,
orbital information, active-orbital indices or explicit active-frame
provenance, and the reference determinant, and can be loaded with
`allow_pickle=False`. Natural-orbital inputs additionally contain their
rotation, occupations, active-orbital coefficients, and CASCI density data.
The reconstructed Fe2S2 records retain exact coordinates and active-space
projector-frame provenance; the historical Fe2S2 archives retain their
qualified point identifier because the parent FCIDUMP has no coordinates.
