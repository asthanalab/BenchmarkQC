from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pyscf import fci
from pyscf.fci import spin_op

from benchmark_qc.hamiltonian_test import load_hamiltonian_npz
from benchmark_qc.integral_dataset import (
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    max_pauli_coefficient_difference,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "benchmarkQC" / "source" / "molecules" / "C2" / "cas8e8o_augccpvtz"
METADATA = json.loads((DATASET_ROOT / "metadata.json").read_text(encoding="utf-8"))
VARIANT = METADATA["variants"]["casci_natural_orbitals"]


def target_singlet_properties(archive) -> tuple[float, float, float, bool]:
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


def test_c2_archives_are_complete_and_match_metadata() -> None:
    output = DATASET_ROOT / VARIANT["output"]["path"]
    data = load_hamiltonian_npz(str(output))
    assert sha256_file(output) == VARIANT["output"]["sha256"]
    assert list(np.asarray(data.labels, dtype=float)) == pytest.approx([2.2], abs=1e-12)
    assert list(data.ref_energies) == pytest.approx(
        VARIANT["casci_energies_hartree"], abs=1e-12
    )
    assert [len(terms) for terms in data.hs] == VARIANT["pauli_term_counts"]

    source = VARIANT["source_integral_archives"][0]
    path = DATASET_ROOT / source["path"]
    assert sha256_file(path) == source["sha256"]
    archive = load_spatial_integral_archive(path)
    assert archive.one_body_integrals.shape == (8, 8)
    assert archive.two_body_integrals.shape == (8, 8, 8, 8)
    assert archive.n_qubits == 16
    assert np.array_equal(archive.active_mo_indices, np.arange(2, 10))
    assert int(np.count_nonzero(archive.reference_determinant)) == 8
    bond = np.linalg.norm(archive.geometry_angstrom[1] - archive.geometry_angstrom[0])
    assert bond == pytest.approx(2.2, abs=1e-12)


def test_c2_natural_orbital_provenance_is_self_consistent() -> None:
    path = DATASET_ROOT / VARIANT["source_integral_archives"][0]["path"]
    with np.load(path, allow_pickle=False) as data:
        rotation = np.asarray(data["natural_orbital_rotation"], dtype=float)
        occupations = np.asarray(data["natural_orbital_occupations"], dtype=float)
        source_coeff = np.asarray(data["source_active_hf_mo_coeff"], dtype=float)
        natural_coeff = np.asarray(data["natural_active_mo_coeff"], dtype=float)
        source_rdm1 = np.asarray(data["source_casci_rdm1"], dtype=float)
        natural_rdm1 = np.asarray(
            data["source_casci_rdm1_in_natural_basis"], dtype=float
        )
        assert bool(data["exact_target_information_used"])

    assert np.linalg.norm(rotation.T @ rotation - np.eye(8)) <= 1e-12
    assert natural_coeff == pytest.approx(source_coeff @ rotation, abs=1e-12)
    assert source_rdm1 == pytest.approx(
        rotation @ np.diag(occupations) @ rotation.T, abs=1e-12
    )
    assert natural_rdm1 == pytest.approx(np.diag(occupations), abs=1e-12)
    assert float(np.sum(occupations)) == pytest.approx(8.0, abs=1e-12)


def test_saved_c2_terms_match_numeric_source() -> None:
    output = DATASET_ROOT / VARIANT["output"]["path"]
    data = load_hamiltonian_npz(str(output))
    source = load_spatial_integral_archive(
        DATASET_ROOT / VARIANT["source_integral_archives"][0]["path"]
    )
    rebuilt = jordan_wigner_terms_from_integrals(source, cutoff=1e-14)
    assert max_pauli_coefficient_difference(data.hs[0], rebuilt) <= 1e-12


def test_c2_reference_in_target_spin_sector() -> None:
    output = DATASET_ROOT / VARIANT["output"]["path"]
    data = load_hamiltonian_npz(str(output))
    source = load_spatial_integral_archive(
        DATASET_ROOT / VARIANT["source_integral_archives"][0]["path"]
    )
    energy, residual, spin_square, converged = target_singlet_properties(source)
    assert converged
    assert energy == pytest.approx(float(data.ref_energies[0]), abs=1e-8)
    assert residual <= 1e-9
    assert abs(spin_square) <= 1e-8
