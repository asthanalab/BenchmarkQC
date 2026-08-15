from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmark_qc.hamiltonian_test import infer_n_qubits, load_hamiltonian_npz
from benchmark_qc.molvqe21 import list_cases, load_metadata, load_source_integrals


ROOT = Path(__file__).resolve().parents[1]


def test_molvqe21_has_27_corrected_cases() -> None:
    cases = list_cases()
    assert len(cases) == 27
    assert len({case.case_id for case in cases}) == 27
    assert all(case.hamiltonian_path.is_file() for case in cases)
    assert all(case.integrals_path.is_file() for case in cases)
    assert all(case.metadata_path.is_file() for case in cases)


def test_molvqe21_archives_match_integral_metadata() -> None:
    with (ROOT / "datasets" / "molvqe21" / "source" / "manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        hamiltonian = load_hamiltonian_npz(str(ROOT / "datasets" / "molvqe21" / row["hamiltonian_path"]))
        assert hamiltonian.labels.tolist() == [0.0]
        assert infer_n_qubits(hamiltonian.hs[0]) == int(row["qubits"])
        assert np.isfinite(hamiltonian.ref_energies[0])

        archive = load_source_integrals(row["case_id"])
        assert archive.n_spatial_orbitals == int(row["active_spatial_orbitals"])
        assert archive.n_qubits == int(row["qubits"])
        assert archive.reference_determinant.shape == (int(row["qubits"]),)


def test_molvqe21_metadata_and_catalog_have_same_case_ids() -> None:
    metadata = json.loads((ROOT / "datasets" / "molvqe21" / "metadata.json").read_text(encoding="utf-8"))
    manifest_ids = {case.case_id for case in list_cases()}
    assert set(metadata["cases"]) == manifest_ids

    catalog = json.loads((ROOT / "datasets" / "catalog.json").read_text(encoding="utf-8"))
    catalog_ids = {
        entry["source_case_id"]
        for entry in catalog["datasets"]
        if entry["system"] == "MolVQE21"
    }
    assert catalog_ids == manifest_ids


def test_molvqe21_per_system_metadata_uses_the_shared_contract() -> None:
    for case in list_cases():
        metadata = load_metadata(case.case_id)
        assert metadata["schema"] == "benchmark-qc.system-metadata.v1"
        assert metadata["system"] == "MolVQE21"
        assert metadata["system_id"] == case.case_id
        assert metadata["hamiltonian"]["path"] == (
            f"datasets/molvqe21/systems/{case.case_id}/hamiltonian.npz"
        )
        assert metadata["integrals"]["path"] == (
            f"datasets/molvqe21/systems/{case.case_id}/inputs/source_integrals.npz"
        )
