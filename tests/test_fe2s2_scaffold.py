from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_qc.fe2s2 import (
    CandidateNotAcceptedError,
    HISTORICAL_SOURCE_PATH,
    HISTORICAL_SOURCE_SHA256,
    PARENT_FCIDUMP_PATH,
    PARENT_FCIDUMP_SHA256,
    PUBLISHED_CAS6_ENERGY_HARTREE,
    accepted_variant_available,
    load_hamiltonian,
    load_source_integrals,
    require_accepted_artifacts,
    require_accepted_report,
    require_nested_cross_version_validation,
    sha256_file,
)
from benchmark_qc.integral_dataset import (
    jordan_wigner_terms_from_integrals,
    max_pauli_coefficient_difference,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "datasets" / "benchmarkQC" / "Fe2S2" / "cas6e6o_chan30e20o"


def accepted_report() -> dict:
    return {
        "schema": "benchmark-qc.fe2s2-candidate.v1",
        "dataset_id": "fe2s2_chan30e20o_historical_rhf_cas6e6o",
        "variant": "historical_cas6e6o",
        "status": "accepted",
        "acceptance": {"cas6_energy_tolerance_hartree": 1.0e-6},
        "results": {"casci_energy_hartree": -116.15625982795386},
        "gates": {"source": True, "physics": True},
        "runtime": {"execution_site": "desktop"},
    }


def test_parent_and_historical_source_are_checksum_pinned() -> None:
    assert sha256_file(PARENT_FCIDUMP_PATH) == PARENT_FCIDUMP_SHA256
    assert sha256_file(HISTORICAL_SOURCE_PATH) == HISTORICAL_SOURCE_SHA256
    first_line = PARENT_FCIDUMP_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert "NORB=  20" in first_line
    assert "NELEC=30" in first_line
    assert "MS2=0" in first_line


def test_metadata_template_is_portable_and_not_prematurely_accepted() -> None:
    path = MODEL / "metadata.template.json"
    text = path.read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert metadata["accepted_variants"] == {}
    assert metadata["parent_hamiltonian"]["sha256"] == PARENT_FCIDUMP_SHA256
    assert metadata["target_models"]["historical_cas6e6o"]["published_casci_energy_hartree"] == (
        PUBLISHED_CAS6_ENERGY_HARTREE
    )
    assert {
        metadata["target_models"][name]["published_casci_energy_hartree"]
        for name in ("historical_cas4e4o", "historical_cas6e6o", "historical_cas8e6o")
    } == {-116.154749, -116.156259, -116.158572}
    assert "/Users/" not in text
    assert "OneDrive" not in text


def test_historical_and_asf_routes_are_explicitly_separate() -> None:
    historical_path = MODEL / "reproduce_historical_cas6e6o.py"
    asf_path = MODEL / "generate_asf_cas6e6o.py"
    historical_tree = ast.parse(historical_path.read_text(encoding="utf-8"))
    historical_imports = {
        alias.name
        for node in ast.walk(historical_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "asf" or name.startswith("asf.") for name in historical_imports)
    asf_text = asf_path.read_text(encoding="utf-8")
    assert "find_one_sized" in asf_text
    assert "fallback_unfiltered=False" in asf_text
    assert "fallback_unfiltered=True" in asf_text
    assert "--fallback-unfiltered" in asf_text
    assert "explicit_unfiltered_fallback" in asf_text
    assert "PUBLISHED_CAS6_ENERGY_HARTREE" in asf_text
    assert "execution-site" in asf_text


def test_nested_cas8e8o_extension_has_exact_parent_span_and_no_paper_fit() -> None:
    path = MODEL / "generate_nested_cas8e8o.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"ACTIVE_ELECTRONS", "ACTIVE_ORBITALS", "VARIANT", "DATASET_ID"}
    }
    assert assignments == {
        "ACTIVE_ELECTRONS": 8,
        "ACTIVE_ORBITALS": 8,
        "VARIANT": "nested_cas8e8o",
        "DATASET_ID": "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o",
    }
    assert "EXPECTED_ACTIVE_INDICES = np.arange(11, 19" in text
    assert "HISTORICAL_CAS6_INDICES = np.arange(12, 18" in text
    assert '"published_energy_target_used": False' in text
    assert "physical_candidate_gates" in text
    assert "from workflow_common import" in text
    assert "    candidate_gates," not in text
    assert "historical_paper_space\": False" in text
    assert "--runtime-role" in text
    assert "primary_pyscf2p4" in text
    assert "crosscheck_pyscf2p11" in text
    assert "historical_rhf_mo_fingerprint_policy" in text
    assert "require_exact_mo_coefficients_sha256" in text
    assert "historical_cas6_control_energy" in text
    assert "execution-site" in text


def test_source_archive_keeps_active_and_parent_indices_distinct() -> None:
    common_text = (MODEL / "workflow_common.py").read_text(encoding="utf-8")
    assert "active_mo_indices=np.arange(len(active_indices)" in common_text
    assert "parent_active_mo_indices=np.asarray(active_indices" in common_text


def test_acceptance_helper_rejects_nonaccepted_or_failed_reports() -> None:
    report = accepted_report()
    require_accepted_report(report)

    rejected = copy.deepcopy(report)
    rejected["status"] = "rejected"
    with pytest.raises(CandidateNotAcceptedError, match="not 'accepted'"):
        require_accepted_report(rejected)

    failed = copy.deepcopy(report)
    failed["gates"]["physics"] = False
    with pytest.raises(CandidateNotAcceptedError, match="failed gates"):
        require_accepted_report(failed)

    local = copy.deepcopy(report)
    local["runtime"]["execution_site"] = "local"
    with pytest.raises(CandidateNotAcceptedError, match="desktop or talon"):
        require_accepted_report(local)

    wrong_energy = copy.deepcopy(report)
    wrong_energy["results"]["casci_energy_hartree"] += 1.0e-3
    with pytest.raises(CandidateNotAcceptedError, match="does not reproduce"):
        require_accepted_report(wrong_energy)

    nonfinite = copy.deepcopy(report)
    nonfinite["results"]["casci_energy_hartree"] = float("nan")
    with pytest.raises(CandidateNotAcceptedError, match="does not reproduce"):
        require_accepted_report(nonfinite)

    loose_tolerance = copy.deepcopy(report)
    loose_tolerance["acceptance"]["cas6_energy_tolerance_hartree"] = 1.0
    with pytest.raises(CandidateNotAcceptedError, match="does not reproduce"):
        require_accepted_report(loose_tolerance)


@pytest.mark.parametrize(
    ("variant", "dataset_id", "energy"),
    (
        ("historical_cas4e4o", "fe2s2_chan30e20o_historical_rhf_cas4e4o", -116.1547494),
        ("historical_cas6e6o", "fe2s2_chan30e20o_historical_rhf_cas6e6o", -116.1562598),
        ("historical_cas8e6o", "fe2s2_chan30e20o_historical_rhf_cas8e6o", -116.1585724),
    ),
)
def test_all_historical_sibling_reports_have_separate_targets(
    variant: str, dataset_id: str, energy: float
) -> None:
    report = accepted_report()
    report["variant"] = variant
    report["dataset_id"] = dataset_id
    report["results"]["casci_energy_hartree"] = energy
    require_accepted_report(report)


def test_acceptance_helper_verifies_relative_artifact_checksums(tmp_path) -> None:
    source = tmp_path / "source_integrals.npz"
    hamiltonian = tmp_path / "Fe2S2_H.npz"
    source.write_bytes(b"source")
    hamiltonian.write_bytes(b"hamiltonian")
    report = accepted_report()
    report["artifacts"] = {
        "source_integrals": {
            "path": source.name,
            "sha256": sha256_file(source),
        },
        "hamiltonian": {
            "path": hamiltonian.name,
            "sha256": sha256_file(hamiltonian),
        },
    }
    validated = require_accepted_artifacts(report, report_directory=tmp_path)
    assert validated == {"source_integrals": source, "hamiltonian": hamiltonian}

    hamiltonian.write_bytes(b"changed")
    with pytest.raises(CandidateNotAcceptedError, match="checksum mismatch"):
        require_accepted_artifacts(report, report_directory=tmp_path)

    report["artifacts"]["hamiltonian"]["path"] = "../outside.npz"
    with pytest.raises(CandidateNotAcceptedError, match="safe report-relative"):
        require_accepted_artifacts(report, report_directory=tmp_path)


def test_only_promoted_historical_and_nested_variants_are_exposed() -> None:
    catalog = json.loads((ROOT / "datasets" / "catalog.json").read_text(encoding="utf-8"))
    fe2s2_ids = {
        entry["id"]
        for entry in catalog["datasets"]
        if entry["system"] == "Fe2S2"
    }
    assert fe2s2_ids == {
        "fe2s2_cas10e10o_anorccvdz_mb_published_geometry",
        "fe2s2_cas10e10o_anorccvdz_mb_bridge_stretched_1p10",
        "fe2s2_chan30e20o_historical_rhf_cas4e4o_point0",
        "fe2s2_chan30e20o_historical_rhf_cas6e6o_point0",
        "fe2s2_chan30e20o_historical_rhf_cas8e6o_point0",
        "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o_point0",
    }
    assert accepted_variant_available("historical_cas4e4o")
    assert accepted_variant_available("historical_cas6e6o")
    assert accepted_variant_available("historical_cas8e6o")
    assert accepted_variant_available("nested_cas8e8o")
    assert not accepted_variant_available("asf_dmrg_cas6e6o")


def test_nested_variant_has_cross_version_certificate_and_portable_archives() -> None:
    root = MODEL / "accepted" / "nested_cas8e8o"
    report = json.loads((root / "acceptance_report.json").read_text(encoding="utf-8"))
    require_accepted_artifacts(report, report_directory=root)
    require_nested_cross_version_validation(report, report_directory=root)
    certificate = json.loads(
        (root / "validation" / "cross_version_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert certificate["status"] == "accepted"
    assert certificate["dataset_id"] == report["dataset_id"]
    assert all(certificate["gates"].values())
    for path in (
        root / "inputs" / "source_integrals.npz",
        root / "validation" / "crosscheck_source_integrals.npz",
    ):
        with np.load(path, allow_pickle=False) as archive:
            assert archive["one_body_integrals"].shape == (8, 8)
            assert archive["two_body_integrals"].shape == (8, 8, 8, 8)
            assert archive["parent_active_mo_indices"].tolist() == list(range(11, 19))


def test_nested_cross_version_certificate_binds_promoted_reports(tmp_path) -> None:
    source_root = MODEL / "accepted" / "nested_cas8e8o"
    copied_root = tmp_path / "nested_cas8e8o"
    import shutil

    shutil.copytree(source_root, copied_root)
    report_path = copied_root / "acceptance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require_nested_cross_version_validation(report, report_directory=copied_root)

    crosscheck_path = copied_root / "validation" / "crosscheck_candidate_report.json"
    crosscheck = json.loads(crosscheck_path.read_text(encoding="utf-8"))
    crosscheck["runtime"]["wall_seconds"] = 0.0
    crosscheck_path.write_text(
        json.dumps(crosscheck, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CandidateNotAcceptedError, match="certificate is invalid"):
        require_nested_cross_version_validation(
            report, report_directory=copied_root
        )


def test_nested_hamiltonian_roundtrip_from_pickle_free_integrals() -> None:
    data = load_hamiltonian("nested_cas8e8o")
    source = load_source_integrals("nested_cas8e8o")
    rebuilt = jordan_wigner_terms_from_integrals(source, cutoff=1.0e-14)
    assert max_pauli_coefficient_difference(data.hs[0], rebuilt) <= 1.0e-12
    assert data.ref_energies[0] == pytest.approx(
        -116.16151636450809, abs=1.0e-12
    )
