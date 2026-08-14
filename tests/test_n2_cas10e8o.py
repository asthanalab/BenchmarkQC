from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_qc.hamiltonian_test import ground_energy_from_terms, load_hamiltonian_npz
from benchmark_qc.integral_dataset import (
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    max_pauli_coefficient_difference,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "benchmarkQC" / "N2" / "cas10e8o_ccpvdz"
METADATA = json.loads((DATASET_ROOT / "metadata.json").read_text(encoding="utf-8"))
GEOMETRY_IDS = ("equilibrium_r1p0977", "stretched_r2p0000")


@pytest.mark.parametrize("variant", ("canonical", "casci_natural_orbitals"))
def test_source_archives_are_complete_and_match_metadata(variant: str) -> None:
    variant_metadata = METADATA["variants"][variant]
    output = DATASET_ROOT / variant_metadata["output"]["path"]
    data = load_hamiltonian_npz(str(output))
    assert sha256_file(output) == variant_metadata["output"]["sha256"]
    assert list(np.asarray(data.ref_energies, dtype=float)) == pytest.approx(
        variant_metadata["casci_energies_hartree"], abs=1e-12
    )
    assert [len(terms) for terms in data.hs] == variant_metadata["pauli_term_counts"]
    for source, geometry in zip(variant_metadata["source_integral_archives"], METADATA["geometries"]):
        path = DATASET_ROOT / source["path"]
        assert source["geometry_id"] == geometry["id"]
        assert sha256_file(path) == source["sha256"]
        archive = load_spatial_integral_archive(path)
        assert archive.one_body_integrals.shape == (8, 8)
        assert archive.two_body_integrals.shape == (8, 8, 8, 8)
        assert archive.n_qubits == 16
        assert np.array_equal(archive.active_mo_indices, np.arange(2, 10))
        assert int(np.count_nonzero(archive.reference_determinant)) == 10
        bond = np.linalg.norm(archive.geometry_angstrom[1] - archive.geometry_angstrom[0])
        assert bond == pytest.approx(geometry["bond_length_angstrom"], abs=1e-12)


@pytest.mark.parametrize("variant", ("canonical", "casci_natural_orbitals"))
def test_saved_jordan_wigner_terms_match_numeric_sources(variant: str) -> None:
    data = load_hamiltonian_npz(str(DATASET_ROOT / variant / "N2_PES_H.npz"))
    for index, geometry_id in enumerate(GEOMETRY_IDS):
        source = load_spatial_integral_archive(
            DATASET_ROOT / variant / "inputs" / f"{geometry_id}.npz"
        )
        rebuilt = jordan_wigner_terms_from_integrals(source, cutoff=1e-14)
        assert max_pauli_coefficient_difference(data.hs[index], rebuilt) <= 1e-12


@pytest.mark.parametrize("variant", ("canonical", "casci_natural_orbitals"))
def test_new_n2_references_in_physical_sector(variant: str) -> None:
    data = load_hamiltonian_npz(str(DATASET_ROOT / variant / "N2_PES_H.npz"))
    for index in range(2):
        energy, _ = ground_energy_from_terms(data.hs[index], nelec=10, spin=0)
        assert energy == pytest.approx(float(data.ref_energies[index]), abs=1e-8)
