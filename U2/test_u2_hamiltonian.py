"""Beginner-friendly sanity check for the saved U2 Hamiltonian.

This folder contains a saved PES file with:
- `labels`: scanned geometry labels (e.g., bond length)
- `Hs`: the corresponding *qubit Hamiltonians* (PennyLane operators)
- `casci_energies`: reference energies from the same active-space CASCI/FCI

This script verifies (for one chosen point) that:
- If we diagonalize the saved qubit Hamiltonian, its ground-state energy matches
  the stored reference energy (within a tolerance).

Run examples:
  python U2/test_u2_hamiltonian.py --index 0
  python U2/test_u2_hamiltonian.py --bond 2.48

Notes:
- For U2 here the Hamiltonian is 12 qubits (dimension 4096), so we use a sparse
  eigensolver by default to avoid building a 4096x4096 dense matrix.

Docs:
- docs/USAGE.md
- docs/NPZ_FORMAT.md
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path


# Make repo root importable even when running from inside the U2 folder.
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
        default="U2/U2_PES_H.npz",
        help="Path to the saved PES npz (default: U2/U2_PES_H.npz)",
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

    args = parser.parse_args()

    data = load_hamiltonian_npz(args.npz)

    i = pick_point_index(data.labels, index=args.index, bond=args.bond)
    chosen_label = float(data.labels[i])

    terms = data.hs[i]
    e0, method = ground_energy_from_terms(terms)

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
