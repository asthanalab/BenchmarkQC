from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_qc.hamiltonian_test import infer_n_qubits, load_hamiltonian_npz
from benchmark_qc.integral_dataset import load_spatial_integral_archive


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads((ROOT / "datasets" / "catalog.json").read_text(encoding="utf-8"))


def test_catalog_is_unique_and_scoped_to_checked_in_systems(catalog: dict) -> None:
    entries = catalog["datasets"]
    assert catalog["schema"] == "benchmark-qc.catalog.v2"
    assert len(entries) == 78
    assert sum(entry["system"] != "MolVQE21" for entry in entries) == 51
    assert sum(entry["system"] == "MolVQE21" for entry in entries) == 27
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert {entry["system"] for entry in entries} == {
        "C2",
        "C2H4",
        "CH2",
        "Fe2S2",
        "FeH",
        "FeS",
        "MolVQE21",
        "N2",
        "O2",
        "U2",
    }


@pytest.mark.parametrize("required_key", ("id", "system", "status", "data_path", "sha256"))
def test_every_catalog_entry_has_required_field(catalog: dict, required_key: str) -> None:
    assert all(required_key in entry for entry in catalog["datasets"])


def test_benchmarkqc_and_molvqe21_use_the_same_system_layout(catalog: dict) -> None:
    metadata_key_sets: set[tuple[str, ...]] = set()
    required_files = {
        "hamiltonian.npz",
        "inputs/source_integrals.npz",
        "metadata.json",
    }

    for entry in catalog["datasets"]:
        family = "molvqe21" if entry["system"] == "MolVQE21" else "benchmarkQC"
        case_id = entry.get("source_case_id", entry["id"])
        system_dir = ROOT / "datasets" / family / "systems" / case_id
        assert system_dir.is_dir(), entry["id"]
        assert (ROOT / entry["data_path"]) == system_dir / "hamiltonian.npz"
        assert (ROOT / entry["integral_data_path"]) == system_dir / "inputs" / "source_integrals.npz"
        assert (ROOT / entry["metadata_path"]) == system_dir / "metadata.json"

        files = {
            path.relative_to(system_dir).as_posix()
            for path in system_dir.rglob("*")
            if path.is_file()
        }
        assert files == required_files, entry["id"]

        metadata = json.loads((system_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata_key_sets.add(tuple(sorted(metadata)))
        assert metadata["schema"] == "benchmark-qc.system-metadata.v1"
        assert metadata["system_id"] == case_id
        assert metadata["hamiltonian"]["path"] == entry["data_path"]
        assert metadata["integrals"]["path"] == entry["integral_data_path"]

    assert len(metadata_key_sets) == 1


def test_benchmarkqc_and_molvqe21_have_the_same_family_root_layout() -> None:
    expected_directories = {"systems", "reference", "source"}
    expected_files = {"README.md", "metadata.json"}
    expected_reference_files = {"reference_results.json", "inventory.json"}

    manifests = []
    for family in ("benchmarkQC", "molvqe21"):
        family_root = ROOT / "datasets" / family
        assert {path.name for path in family_root.iterdir() if path.is_dir()} == expected_directories
        assert expected_files == {
            path.name for path in family_root.iterdir() if path.is_file()
        }
        assert {
            path.name for path in (family_root / "reference").iterdir() if path.is_file()
        } == expected_reference_files
        assert (family_root / "source" / "manifest.csv").is_file()
        manifests.append(
            (family_root / "source" / "manifest.csv").read_text(encoding="utf-8").splitlines()[0]
        )

        metadata = json.loads((family_root / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["systems_root"] == "systems"
        assert metadata["manifest_path"] == "source/manifest.csv"
        assert metadata["reference_results_path"] == "reference/reference_results.json"
        assert metadata["inventory_path"] == "reference/inventory.json"

    assert manifests[0] == manifests[1]


def test_catalog_archives_have_exact_schema_and_checksums(catalog: dict) -> None:
    for entry in catalog["datasets"]:
        path = ROOT / entry["data_path"]
        assert path.is_file(), path
        assert file_sha256(path) == entry["sha256"]
        with np.load(path, allow_pickle=True) as raw:
            assert set(raw.files) == {"labels", "Hs", "casci_energies"}
            assert raw["labels"].shape == (entry["point_count"],)
            assert raw["labels"].dtype == object
            assert raw["Hs"].shape == (entry["point_count"],)
            assert raw["Hs"].dtype == object
            assert raw["casci_energies"].shape == (entry["point_count"],)
            assert raw["casci_energies"].dtype == np.float64
            assert np.all(np.isfinite(np.asarray(raw["labels"], dtype=float)))
            assert np.all(np.isfinite(raw["casci_energies"]))

        data = load_hamiltonian_npz(str(path))
        assert all(infer_n_qubits(terms) == entry["qubits"] for terms in data.hs)
        metadata_path = entry.get("metadata_path")
        if metadata_path is not None:
            assert (ROOT / metadata_path).is_file()


def test_every_catalog_entry_has_a_validated_normalized_integral_archive(catalog: dict) -> None:
    """Every point-level system must expose the same numeric input contract."""

    entries = catalog["datasets"]
    assert len(entries) == 78
    for entry in entries:
        integral_path = entry.get("integral_data_path")
        integral_sha256 = entry.get("integral_sha256")
        assert integral_path, entry["id"]
        assert integral_sha256, entry["id"]
        archive_path = ROOT / integral_path
        assert archive_path.is_file(), entry["id"]
        assert file_sha256(archive_path) == integral_sha256
        archive = load_spatial_integral_archive(archive_path)
        assert archive.n_qubits == entry["qubits"]
        assert archive.one_body_integrals.shape == (
            entry["active_spatial_orbitals"],
            entry["active_spatial_orbitals"],
        )


def test_every_benchmarkqc_point_has_common_scalar_reference_results(catalog: dict) -> None:
    results_path = ROOT / "datasets" / "benchmarkQC" / "reference" / "reference_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["schema"] == "benchmark-qc.reference-results.v1"
    benchmark_entries = [entry for entry in catalog["datasets"] if entry["system"] != "MolVQE21"]
    assert results["case_count"] == len(benchmark_entries) == 51
    assert set(results["cases"]) == {entry["id"] for entry in benchmark_entries}
    required = {
        "casci_energy_hartree",
        "cisd_energy_hartree",
        "cisd_status",
        "ccsd_status",
        "renyi_0p25_nats",
        "cumulant_C2",
    }
    for entry in benchmark_entries:
        record = results["cases"][entry["id"]]
        metadata = json.loads((ROOT / entry["metadata_path"]).read_text(encoding="utf-8"))
        assert required.issubset(record)
        assert metadata["reference_results"] == record
        assert np.isfinite(record["casci_energy_hartree"])
        assert np.isfinite(record["cisd_energy_hartree"])
        assert np.isfinite(record["renyi_0p25_nats"])
        assert np.isfinite(record["cumulant_C2"])


def test_si_table_i_inventory_covers_all_rows_after_reconstruction() -> None:
    inventory = json.loads(
        (ROOT / "datasets" / "benchmarkQC" / "source" / "si_table_i_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        geometry
        for variant in inventory["variants"]
        for geometry in variant["geometry_rows"]
    ]
    assert inventory["geometry_row_count"] == 26
    assert len(inventory["variants"]) == 13
    assert len(rows) == 26
    assert sum(row["status"] == "checked_in_payload" for row in rows) == 26
    assert sum(row["status"] == "cloud_placeholder_payload" for row in rows) == 0
    assert sum(row["status"] == "results_metadata_only" for row in rows) == 0
    catalog_ids = {
        entry["id"]
        for entry in json.loads((ROOT / "datasets" / "catalog.json").read_text())[
            "datasets"
        ]
    }
    assert "u2_cas6e10o_anorccvdzp_sfx2c_equilibrium_r2p4300" in catalog_ids
    assert "u2_cas6e10o_anorccvdzp_sfx2c_stretched_r2p8000" in catalog_ids


def test_reconstructed_si_metadata_records_passed_validation() -> None:
    inventory = json.loads(
        (ROOT / "datasets" / "benchmarkQC" / "source" / "si_table_i_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    for variant in inventory["variants"]:
        for row in variant["geometry_rows"]:
            source_archive = row.get("source_archive")
            if source_archive is None or "/systems/" not in source_archive:
                continue
            metadata_path = ROOT / source_archive.replace("hamiltonian.npz", "metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                assert metadata["si_validation"]["passed"] is True


def test_new_n2_metadata_is_path_portable() -> None:
    path = ROOT / "datasets" / "benchmarkQC" / "source" / "molecules" / "N2" / "cas10e8o_ccpvdz" / "metadata.json"
    text = path.read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert metadata["system"]["formula"] == "N2"
    assert set(metadata["variants"]) == {"canonical", "casci_natural_orbitals"}
    assert "/Users/" not in text
    assert "OneDrive" not in text


def test_c2_metadata_is_path_portable_and_marks_exact_orbital_information() -> None:
    path = ROOT / "datasets" / "benchmarkQC" / "source" / "molecules" / "C2" / "cas8e8o_augccpvtz" / "metadata.json"
    text = path.read_text(encoding="utf-8")
    metadata = json.loads(text)
    variant = metadata["variants"]["casci_natural_orbitals"]
    assert metadata["system"]["formula"] == "C2"
    assert variant["exact_target_information_used"] is True
    assert "/Users/" not in text
    assert "OneDrive" not in text


def test_fe2s2_metadata_is_portable_and_discloses_historical_frame() -> None:
    path = ROOT / "datasets" / "benchmarkQC" / "source" / "molecules" / "Fe2S2" / "cas6e6o_chan30e20o" / "metadata.json"
    text = path.read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert set(metadata["accepted_variants"]) == {
        "historical_cas4e4o",
        "historical_cas6e6o",
        "historical_cas8e6o",
        "nested_cas8e8o",
    }
    for record in metadata["accepted_variants"].values():
        assert record["method"]["stationary_rhf_claim"] is False
        assert record["method"]["active_space_finder_used"] is False
    workflow_status = {
        name: record["status"]
        for name, record in metadata["candidate_variants"].items()
    }
    assert workflow_status == {
        "asf_dmrg_cas6e6o": "not yet accepted",
        "historical_cas4e4o": "accepted",
        "historical_cas6e6o": "accepted",
        "historical_cas8e6o": "accepted",
        "nested_cas8e8o": "accepted",
    }
    assert "/Users/" not in text
    assert "OneDrive" not in text


def test_molvqe21_metadata_is_portable_and_excludes_incomplete_source_cases() -> None:
    path = ROOT / "datasets" / "molvqe21" / "metadata.json"
    text = path.read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert metadata["source_case_count"] == 27
    assert metadata["benchmark_ready_case_count"] == 27
    assert len(metadata["cases"]) == 27
    assert "/Users/" not in text
    assert "OneDrive" not in text

    overrides = json.loads(
        (ROOT / "datasets" / "molvqe21" / "source" / "active_orbital_overrides.json").read_text(
            encoding="utf-8"
        )
    )
    assert overrides["cyclobutadiene_ceo"]["active_orbital_indices"] == [11, 13, 14, 19]
    assert overrides["m_Benzyne_equi_cc_pVTZ_0"]["active_orbital_indices"] == [15, 18, 19, 20, 24, 27]
    assert overrides["m_Benzyne_equi_cc_pVTZ_2"]["active_orbital_indices"] == [16, 18, 19, 20, 24, 27]


def test_molvqe21_source_manifest_contains_only_corrected_cache_cases() -> None:
    path = ROOT / "datasets" / "molvqe21" / "source" / "source_manifest.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = {row["case_id"] for row in rows}
    assert len(rows) == 27
    assert "Ferrocene_ceo" not in case_ids
    assert "feo4_2minus_ceo" not in case_ids
