"""Beginner-friendly sanity check for the saved N2 Hamiltonian.

This repo stores, for each bond length, a *qubit Hamiltonian* (PennyLane operator)
and a reference energy. This script verifies that the stored Hamiltonian produces
the same ground-state energy when we diagonalize it exactly.

What this script does (mirrors the notebook):
1) Load the npz file (contains: labels, Hs, casci_energies)
2) Choose one point by index OR by bond length (nearest match)
3) Convert the stored Pauli-term representation into a dense matrix
4) Compute the ground-state energy by exact diagonalization
5) Compare to the stored reference energy

Run examples:
  python N2/test_n2_hamiltonian.py --index 0
  python N2/test_n2_hamiltonian.py --bond 1.4

Interpretation:
- The diagonalization energy is the *exact* ground energy of the saved qubit Hamiltonian.
- The stored `casci_energies[i]` should match within numerical tolerance.

Docs:
- docs/USAGE.md
- docs/NPZ_FORMAT.md
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path


# Make repo root importable even when running from inside the N2 folder.
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
        default="N2/N2_PES_H1.npz",
        help="Path to the saved PES npz (default: N2/N2_PES_H1.npz)",
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

    # -------------------- Load the file (same as notebook) --------------------
    data = load_hamiltonian_npz(args.npz)

    # -------------------- Choose which geometry point ------------------------
    i = pick_point_index(data.labels, index=args.index, bond=args.bond)
    chosen_label = float(data.labels[i])

    # -------------------- Pull out the saved Hamiltonian ---------------------
    # `hs[i]` is an array of PennyLane terms (object dtype).
    terms = data.hs[i]

    # -------------------- Diagonalize (exact for this saved qubit Hamiltonian) ------
    e0, method = ground_energy_from_terms(terms)

    # -------------------- Compare to stored reference energy -----------------
    e_ref = float(data.ref_energies[i])
    abs_diff = abs(e0 - e_ref)

    # -------------------- Pretty output --------------------------------------
    print(f"NPZ: {args.npz}")
    print(f"Chosen point index:         {i}")
    print(f"Point label (bond length):  {chosen_label}")
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
