#!/usr/bin/env python3
"""Validate the C2 CAS(8e,8o) Hamiltonian for the target spin singlet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from pyscf import fci
from pyscf.fci import spin_op


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_qc.hamiltonian_test import (  # noqa: E402
    load_hamiltonian_npz,
)
from benchmark_qc.integral_dataset import (  # noqa: E402
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    max_pauli_coefficient_difference,
)


def target_singlet_energy(archive) -> tuple[float, float, float, bool]:
    """Return the unpenalized singlet energy, residual, S^2, and convergence."""

    norb = archive.n_spatial_orbitals
    nelec = (4, 4)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-13
    solver.conv_tol_residual = 1e-12
    solver.max_cycle = 1000
    solver.max_space = 100
    solver.davidson_only = True
    solver.lindep = 1e-24
    fci.addons.fix_spin_(solver, shift=0.5, ss=0.0)
    _, vector = solver.kernel(
        archive.one_body_integrals,
        archive.two_body_integrals,
        norb,
        nelec,
        ecore=archive.core_constant,
    )
    effective_h2 = fci.direct_spin1.absorb_h1e(
        archive.one_body_integrals,
        archive.two_body_integrals,
        norb,
        nelec,
        0.5,
    )
    contracted = fci.direct_spin1.contract_2e(effective_h2, vector, norb, nelec)
    norm = float(np.linalg.norm(vector))
    electronic_energy = float(np.vdot(vector, contracted).real / norm**2)
    residual = float(
        np.linalg.norm(contracted - electronic_energy * vector) / norm
    )
    spin_square, _ = spin_op.spin_square0(vector, norb, nelec)
    return (
        archive.core_constant + electronic_energy,
        residual,
        float(spin_square),
        bool(solver.converged),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = HERE / "casci_natural_orbitals" / "C2_PES_H.npz"
    data = load_hamiltonian_npz(str(path))
    if len(data.labels) != 1:
        raise RuntimeError(f"Expected one C2 geometry, found {len(data.labels)}")
    source_path = (
        HERE
        / "casci_natural_orbitals"
        / "inputs"
        / "stretched_r2p2000.npz"
    )
    archive = load_spatial_integral_archive(source_path)
    rebuilt = jordan_wigner_terms_from_integrals(archive, cutoff=1e-14)
    coefficient_difference = max_pauli_coefficient_difference(data.hs[0], rebuilt)
    energy, residual, spin_square, converged = target_singlet_energy(archive)
    reference = float(data.ref_energies[0])
    difference = abs(energy - reference)
    print(f"Dataset: {path.relative_to(REPOSITORY_ROOT)}")
    print(f"R (Angstrom): {float(data.labels[0]):.4f}")
    print("Target: N=8, spin_2S=0")
    print("Solver: PySCF direct_spin1 FCI with S^2=0 penalty")
    print(f"Stored CASCI (Ha): {reference:.12f}")
    print(f"Unpenalized target-singlet energy (Ha): {energy:.12f}")
    print(f"Absolute difference (Ha): {difference:.3e}")
    print(f"Physical-Hamiltonian residual (Ha): {residual:.3e}")
    print(f"S^2 expectation: {spin_square:.3e}")
    print(f"Maximum saved/source Pauli difference: {coefficient_difference:.3e}")
    passed = (
        converged
        and difference <= args.atol
        and residual <= 1e-9
        and abs(spin_square) <= 1e-8
        and coefficient_difference <= 1e-12
    )
    if passed:
        print(f"PASS (difference <= {args.atol:.1e} Ha)")
        return 0
    print(f"FAIL (difference > {args.atol:.1e} Ha)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
