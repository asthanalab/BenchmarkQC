# Saved NPZ Format

The PES `.npz` files (e.g. `N2/N2_PES_H1.npz`, `FeS/FeS_PES_H.npz`, `U2/U2_PES_H.npz`) share the same format.

They contain three arrays:

## `labels`

- dtype: `object`
- shape: `(n_points,)`
- meaning: the scanned geometry label for each point.

In this repo’s provided notebooks, this is the **bond length in Angstrom** (a float).
This follows PySCF’s default unit (the generators do not override `mol.unit`).

For the exact grids used by each system (N2/FeS/U2), see `docs/USAGE.md` →
“Stored geometry points (bond lengths)”.

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
- For larger ones (e.g. FeS/U2: 12 qubits → 4096×4096), tests use sparse eigen-solvers.
- For open-shell/high-spin systems (FeS), you must compare in the same `(N_alpha, N_beta)` sector.
  The FeS test does this using `--nelec` and `--spin`.
