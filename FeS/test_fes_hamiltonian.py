"""Beginner-friendly sanity check for the saved FeS Hamiltonian.

This folder contains a saved PES file with:
- `labels`: the scanned geometry parameter(s) (often the bond length)
- `Hs`: the corresponding *qubit Hamiltonians* (PennyLane operators)
- `casci_energies`: reference energies from the same active-space CASCI/FCI

This script verifies (for one chosen point) that:
- If we diagonalize the saved qubit Hamiltonian, its ground-state energy matches
  the stored reference energy (within a tolerance).

Run examples:
  python FeS/test_fes_hamiltonian.py --index 0
  python FeS/test_fes_hamiltonian.py --bond 2.4

Notes:
- For FeS here the Hamiltonian is 12 qubits (dimension 4096), so we use a sparse
  eigensolver by default to avoid building a 4096x4096 dense matrix.
- FeS is an open-shell high-spin state. To compare apples-to-apples against the
    stored CASCI energy, we must diagonalize *within the same (N, M_S) sector*.
    This script does that using `--nelec` and `--spin` (PySCF convention: spin = 2S).

Docs:
- docs/USAGE.md
- docs/NPZ_FORMAT.md
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path


# Make repo root importable even when running from inside the FeS folder.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_qc.hamiltonian_test import (  # noqa: E402
    ground_energy_from_terms,
    load_hamiltonian_npz,
    pick_point_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="FeS/FeS_PES_H.npz",
        help="Path to the saved PES npz (default: FeS/FeS_PES_H.npz)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Point index in labels/Hs arrays")
    group.add_argument(
        "--bond",
        type=float,
        help="Bond length to test (chooses nearest stored label)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for energy match (Ha)",
    )
    parser.add_argument(
        "--nelec",
        type=int,
        default=6,
        help="Active-space electrons for the comparison sector (default: 6, from FeS-in.ipynb)",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=4,
        help="PySCF spin (2S) for the comparison sector (default: 4, from FeS-in.ipynb)",
    )

    args = parser.parse_args()

    # Load exactly like the notebooks do
    data = load_hamiltonian_npz(args.npz)

    i = pick_point_index(data.labels, index=args.index, bond=args.bond)
    chosen_label = float(data.labels[i])

    terms = data.hs[i]
    # IMPORTANT for open-shell systems: compare in the same (nelec, spin) sector.
    e0, method = ground_energy_from_terms(terms, nelec=args.nelec, spin=args.spin)

    e_ref = float(data.ref_energies[i])
    abs_diff = abs(e0 - e_ref)

    print(f"NPZ: {args.npz}")
    print(f"Chosen point index:         {i}")
    print(f"Point label:                {chosen_label}")
    print(f"Diagonalization method:     {method}")
    print(f"Stored CASCI energy (Ha):   {e_ref:.12f}")
    print(f"Diag ground energy (Ha):    {e0:.12f}")
    print(f"|diff| (Ha):                {abs_diff:.6e}")

    if abs_diff <= args.atol:
        print(f"PASS (|diff| <= {args.atol})")
        return 0

    print(f"FAIL (|diff| > {args.atol})")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
