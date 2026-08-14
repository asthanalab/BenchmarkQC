#!/usr/bin/env python3
"""Validate one N2 CAS(10e,8o)/cc-pVDZ Hamiltonian in its physical sector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_qc.hamiltonian_test import (  # noqa: E402
    ground_energy_from_terms,
    load_hamiltonian_npz,
    pick_point_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("canonical", "casci_natural_orbitals"),
        default="casci_natural_orbitals",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--bond", type=float)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = HERE / args.variant / "N2_PES_H.npz"
    data = load_hamiltonian_npz(str(path))
    index = pick_point_index(data.labels, index=args.index, bond=args.bond)
    energy, solver = ground_energy_from_terms(data.hs[index], nelec=10, spin=0)
    reference = float(data.ref_energies[index])
    difference = abs(energy - reference)
    print(f"Dataset: {path.relative_to(REPOSITORY_ROOT)}")
    print(f"R (Angstrom): {float(data.labels[index]):.4f}")
    print(f"Solver: {solver}")
    print(f"Stored CASCI (Ha): {reference:.12f}")
    print(f"Sector ground energy (Ha): {energy:.12f}")
    print(f"Absolute difference (Ha): {difference:.3e}")
    if difference <= args.atol:
        print(f"PASS (difference <= {args.atol:.1e} Ha)")
        return 0
    print(f"FAIL (difference > {args.atol:.1e} Ha)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
