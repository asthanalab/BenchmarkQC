#!/usr/bin/env python3
"""Validate and optionally promote one accepted Fe2S2 candidate.

Validation is the default and does not modify the repository.  ``--promote``
copies checksum-verified files into ``accepted/<variant>``, writes the accepted
metadata record, and only then adds the archive to ``datasets/catalog.json``.  Rejected
or incomplete candidates can never reach the root catalog through this tool.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_qc.fe2s2 import (  # noqa: E402
    ACCEPTED_VARIANTS,
    require_accepted_artifacts,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument(
        "--crosscheck-report",
        type=Path,
        help="Required accepted PySCF 2.11 report for nested_cas8e8o.",
    )
    parser.add_argument(
        "--cross-version-certificate",
        type=Path,
        help="Required accepted cross-version certificate for nested_cas8e8o.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Perform the repository write after validation; otherwise only audit.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def catalog_entry(report: dict, hamiltonian_path: Path) -> dict:
    relative = hamiltonian_path.relative_to(REPOSITORY_ROOT)
    target = report["target"]
    entry = {
        "id": report["dataset_id"],
        "system": "Fe2S2",
        "status": "current",
        "data_path": str(relative),
        "metadata_path": str((HERE / "metadata.json").relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(hamiltonian_path),
        "point_count": 1,
        "basis": "inherited from checksum-pinned Li-Chan FCIDUMP; AO basis not encoded",
        "active_electrons": int(target["active_electrons"]),
        "active_spatial_orbitals": int(target["active_spatial_orbitals"]),
        "qubits": int(target["qubits"]),
        "spin_2S": int(target["spin_2S"]),
        "coverage_note": (
            "One archived [Fe2S2(SCH3)4]2- geometry; label 0.0 is a point "
            "identifier because coordinates are absent from the FCIDUMP"
        ),
    }
    if report["variant"].startswith("historical_"):
        entry.update(
            {
                "orbital_frame_status": "historical-nonstationary-default-RHF",
                "active_space_finder_used": False,
                "claim_boundary": (
                    "Faithful paper-Hamiltonian reproduction; not a conventional "
                    "stationary-RHF or ASF-selected frame"
                ),
            }
        )
    elif report["variant"] == "nested_cas8e8o":
        entry.update(
            {
                "orbital_frame_status": "historical-nonstationary-default-RHF",
                "active_space_finder_used": False,
                "historical_paper_space": False,
                "claim_boundary": (
                    "Current-project same-protocol nested CAS(8e,8o) extension; "
                    "not one of the historical paper active spaces"
                ),
            }
        )
    else:
        selection = report["method"]["selection"]
        entry.update(
            {
                "active_space_finder_used": True,
                "asf_selection_mode": selection["selection_mode"],
                "asf_unfiltered_fallback_used": selection[
                    "unfiltered_fallback_used"
                ],
            }
        )
    return entry


def main() -> int:
    args = parse_args()
    report_path = args.candidate_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = require_accepted_artifacts(
        report, report_directory=report_path.parent
    )
    variant = str(report["variant"])
    if variant not in ACCEPTED_VARIANTS:
        raise ValueError(f"unknown accepted variant directory for {variant!r}")
    crosscheck_report_path = None
    crosscheck_report = None
    crosscheck_artifacts = None
    certificate_path = None
    certificate = None
    source_certificate_sha256 = None
    promoted_certificate_sha256 = None
    if variant == "nested_cas8e8o":
        if args.crosscheck_report is None or args.cross_version_certificate is None:
            raise ValueError(
                "nested_cas8e8o requires --crosscheck-report and "
                "--cross-version-certificate"
            )
        if report.get("acceptance", {}).get("runtime_role") != "primary_pyscf2p4":
            raise ValueError("nested promotion requires the PySCF 2.4 primary report")
        crosscheck_report_path = args.crosscheck_report.resolve()
        crosscheck_report = json.loads(
            crosscheck_report_path.read_text(encoding="utf-8")
        )
        crosscheck_artifacts = require_accepted_artifacts(
            crosscheck_report,
            report_directory=crosscheck_report_path.parent,
        )
        if (
            crosscheck_report.get("variant") != variant
            or crosscheck_report.get("acceptance", {}).get("runtime_role")
            != "crosscheck_pyscf2p11"
        ):
            raise ValueError("nested cross-check report has the wrong role or variant")
        certificate_path = args.cross_version_certificate.resolve()
        source_certificate_sha256 = sha256_file(certificate_path)
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        certificate_gates = certificate.get("gates")
        if (
            certificate.get("schema")
            != "benchmark-qc.fe2s2-cross-version-certificate.v1"
            or certificate.get("status") != "accepted"
            or certificate.get("dataset_id") != report["dataset_id"]
            or not isinstance(certificate_gates, dict)
            or not certificate_gates
            or not all(value is True for value in certificate_gates.values())
            or certificate.get("inputs", {})
            .get("primary_report", {})
            .get("sha256")
            != sha256_file(report_path)
            or certificate.get("inputs", {})
            .get("crosscheck_report", {})
            .get("sha256")
            != sha256_file(crosscheck_report_path)
        ):
            raise ValueError("nested cross-version certificate is invalid")

    print(
        json.dumps(
            {
                "validation": "PASS",
                "dataset_id": report["dataset_id"],
                "variant": variant,
                "source_integrals_sha256": sha256_file(artifacts["source_integrals"]),
                "hamiltonian_sha256": sha256_file(artifacts["hamiltonian"]),
                "cross_version_certificate_sha256": (
                    sha256_file(certificate_path) if certificate_path else None
                ),
                "promote_requested": bool(args.promote),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.promote:
        return 0

    destination = ACCEPTED_VARIANTS[variant]
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace an existing accepted variant: {destination}"
        )

    metadata_template = json.loads(
        (HERE / "metadata.template.json").read_text(encoding="utf-8")
    )
    metadata_path = HERE / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else copy.deepcopy(metadata_template)
    )
    metadata["publication_state"] = "one or more checksum-accepted variants cataloged"
    metadata.setdefault("accepted_variants", {})[variant] = {
        "dataset_id": report["dataset_id"],
        "status": "accepted",
        "method": report["method"],
        "acceptance": report["acceptance"],
        "results": report["results"],
        "runtime": report["runtime"],
        "artifact_paths": {
            "source_integrals": f"accepted/{variant}/inputs/source_integrals.npz",
            "hamiltonian": f"accepted/{variant}/Fe2S2_H.npz",
            "acceptance_report": f"accepted/{variant}/acceptance_report.json",
        },
    }
    if variant == "nested_cas8e8o":
        metadata["accepted_variants"][variant]["cross_version_validation"] = {
            "certificate": (
                f"accepted/{variant}/validation/cross_version_certificate.json"
            ),
            "crosscheck_report": (
                f"accepted/{variant}/validation/crosscheck_candidate_report.json"
            ),
            "crosscheck_source_integrals": (
                f"accepted/{variant}/validation/crosscheck_source_integrals.npz"
            ),
            "crosscheck_hamiltonian": (
                f"accepted/{variant}/validation/crosscheck_Fe2S2_H.npz"
            ),
        }
        metadata.setdefault("target_models", {})[variant] = {
            "active_electrons": 8,
            "active_spatial_orbitals": 8,
            "active_spin_orbitals": 16,
            "active_alpha_electrons": 4,
            "active_beta_electrons": 4,
            "spin_2S": 0,
            "published_casci_energy_applicable": False,
            "historical_paper_space": False,
            "parent_active_indices_zero_based": list(range(11, 19)),
        }
        metadata.setdefault("candidate_variants", {})[variant] = {
            "status": "accepted",
            "definition": (
                "Current-project same-protocol nested CAS(8e,8o) extension "
                "of the shared historical default-RHF frame"
            ),
            "orbital_frame_status": "historical nonstationary default-RHF frame",
            "stationary_rhf_claim": False,
            "active_space_finder_used": False,
            "historical_paper_space": False,
            "published_energy_target_used": False,
            "claim_boundary": (
                "This is a validated nested comparison model, not one of the "
                "historical paper active spaces."
            ),
            "source_cross_version_certificate_sha256": source_certificate_sha256,
        }

    catalog_path = REPOSITORY_ROOT / "datasets" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if any(item["id"] == report["dataset_id"] for item in catalog["datasets"]):
        raise RuntimeError(f"catalog already contains {report['dataset_id']}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{variant}.", dir=destination.parent
    ) as temporary_name:
        staged = Path(temporary_name) / variant
        (staged / "inputs").mkdir(parents=True)
        shutil.copy2(artifacts["source_integrals"], staged / "inputs" / "source_integrals.npz")
        shutil.copy2(artifacts["hamiltonian"], staged / "Fe2S2_H.npz")
        promoted_report = copy.deepcopy(report)
        promoted_report["artifacts"]["source_integrals"]["path"] = (
            "inputs/source_integrals.npz"
        )
        promoted_report["artifacts"]["hamiltonian"]["path"] = "Fe2S2_H.npz"
        promoted_report["parent"]["path"] = (
            "../../parent/chan_fe2s2_cas30e20o.FCIDUMP"
        )
        atomic_json(staged / "acceptance_report.json", promoted_report)
        if variant == "nested_cas8e8o":
            assert crosscheck_report is not None
            assert crosscheck_artifacts is not None
            assert certificate is not None
            validation = staged / "validation"
            validation.mkdir()
            shutil.copy2(
                crosscheck_artifacts["source_integrals"],
                validation / "crosscheck_source_integrals.npz",
            )
            shutil.copy2(
                crosscheck_artifacts["hamiltonian"],
                validation / "crosscheck_Fe2S2_H.npz",
            )
            portable_crosscheck = copy.deepcopy(crosscheck_report)
            portable_crosscheck["parent"]["path"] = (
                "../../../parent/chan_fe2s2_cas30e20o.FCIDUMP"
            )
            portable_crosscheck["artifacts"]["source_integrals"]["path"] = (
                "crosscheck_source_integrals.npz"
            )
            portable_crosscheck["artifacts"]["hamiltonian"]["path"] = (
                "crosscheck_Fe2S2_H.npz"
            )
            atomic_json(
                validation / "crosscheck_candidate_report.json",
                portable_crosscheck,
            )
            portable_certificate = copy.deepcopy(certificate)
            portable_certificate["inputs"]["primary_report"]["path"] = (
                "../acceptance_report.json"
            )
            portable_certificate["inputs"]["crosscheck_report"]["path"] = (
                "crosscheck_candidate_report.json"
            )
            portable_certificate["inputs"]["primary_report"]["sha256"] = (
                sha256_file(staged / "acceptance_report.json")
            )
            portable_certificate["inputs"]["crosscheck_report"]["sha256"] = (
                sha256_file(validation / "crosscheck_candidate_report.json")
            )
            atomic_json(
                validation / "cross_version_certificate.json",
                portable_certificate,
            )
            promoted_certificate_sha256 = sha256_file(
                validation / "cross_version_certificate.json"
            )
        os.replace(staged, destination)

    if variant == "nested_cas8e8o":
        metadata["candidate_variants"][variant][
            "promoted_cross_version_certificate_sha256"
        ] = promoted_certificate_sha256
        cross_validation = metadata["accepted_variants"][variant][
            "cross_version_validation"
        ]
        cross_validation["certificate_sha256"] = promoted_certificate_sha256
        cross_validation["crosscheck_report_sha256"] = sha256_file(
            destination / "validation" / "crosscheck_candidate_report.json"
        )

    entry = catalog_entry(report, destination / "Fe2S2_H.npz")
    catalog["datasets"].append(entry)
    catalog["datasets"].sort(key=lambda item: (item["system"], item["id"]))
    atomic_json(metadata_path, metadata)
    atomic_json(catalog_path, catalog)
    print(f"Promoted {report['dataset_id']} and added its verified archive to datasets/catalog.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
