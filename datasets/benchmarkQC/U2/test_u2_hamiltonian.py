"""Beginner-friendly sanity check for the saved U2 Hamiltonian.

This folder contains a saved PES file with:
- `labels`: scanned geometry labels (e.g., bond length)
- `Hs`: the corresponding *qubit Hamiltonians* (PennyLane operators)
- `casci_energies`: reference energies from the same active-space CASCI/FCI

This script verifies (for one chosen point) that:
- If we diagonalize the saved qubit Hamiltonian, its ground-state energy matches
  the stored reference energy (within a tolerance).

Run examples:
  python datasets/benchmarkQC/U2/test_u2_hamiltonian.py --index 0
  python datasets/benchmarkQC/U2/test_u2_hamiltonian.py --bond 2.48

Notes:
- The reference is validated in the six-electron singlet sector. This avoids
  comparing against an unphysical particle-number sector of the 12-qubit Hamiltonian.

Docs:
- docs/USAGE.md
- docs/NPZ_FORMAT.md
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path


# Make the src-layout package importable from a notebook launched in this folder.
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from benchmark_qc.hamiltonian_test import (  # noqa: E402
    ground_energy_from_terms,
    load_hamiltonian_npz,
    pick_point_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="datasets/benchmarkQC/U2/U2_PES_H.npz",
        help="Path to the saved PES npz (default: datasets/benchmarkQC/U2/U2_PES_H.npz)",
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
        help="Active-space electrons for the comparison sector (default: 6)",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="PySCF spin=2S for the comparison sector (default: 0)",
    )

    args = parser.parse_args()

    data = load_hamiltonian_npz(args.npz)

    i = pick_point_index(data.labels, index=args.index, bond=args.bond)
    chosen_label = float(data.labels[i])

    terms = data.hs[i]
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
