#!/usr/bin/env python3
"""Certify cross-version agreement for the derived nested CAS(8e,8o) row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any

import numpy as np


EXPECTED_DATASET_ID = "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o"
EXPECTED_PARENT_SHA256 = (
    "95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9"
)
EXPECTED_ACTIVE_INDICES = np.arange(11, 19, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--crosscheck-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(f"unexpected dataset in {path}")
    if report.get("status") != "accepted":
        raise ValueError(f"report is not accepted: {path}")
    if not report.get("gates") or not all(report["gates"].values()):
        raise ValueError(f"report contains a failed or missing gate: {path}")
    if report.get("parent", {}).get("sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError(f"parent checksum mismatch in {path}")
    return report


def artifact_path(report_path: Path, report: dict[str, Any], key: str) -> Path:
    record = report["artifacts"][key]
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path for {key}")
    path = (report_path.parent / relative).resolve()
    if report_path.parent.resolve() not in path.parents:
        raise ValueError(f"artifact leaves report directory for {key}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact checksum mismatch for {key}: {path}")
    return path


def load_numeric_archive(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = np.asarray(archive[name])
            if value.dtype.kind == "O":
                raise ValueError(f"numeric archive contains object array {name!r}")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError(f"numeric archive contains non-finite values in {name!r}")
            arrays[name] = value
            audit[name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest(),
            }
    return arrays, audit


def scalar_difference(primary: float, crosscheck: float) -> float:
    return abs(float(primary) - float(crosscheck))


def main() -> int:
    args = parse_args()
    hostname = socket.gethostname()
    if not hostname.lower().startswith("undmedaasthanad"):
        raise RuntimeError(f"comparison must run on desktop, got {hostname!r}")

    primary_path = args.primary_report.resolve()
    crosscheck_path = args.crosscheck_report.resolve()
    primary = load_report(primary_path)
    crosscheck = load_report(crosscheck_path)
    if primary["acceptance"]["runtime_role"] != "primary_pyscf2p4":
        raise ValueError("primary report has the wrong runtime role")
    if crosscheck["acceptance"]["runtime_role"] != "crosscheck_pyscf2p11":
        raise ValueError("cross-check report has the wrong runtime role")

    primary_source = artifact_path(primary_path, primary, "source_integrals")
    crosscheck_source = artifact_path(
        crosscheck_path, crosscheck, "source_integrals"
    )
    primary_hamiltonian = artifact_path(primary_path, primary, "hamiltonian")
    crosscheck_hamiltonian = artifact_path(
        crosscheck_path, crosscheck, "hamiltonian"
    )
    primary_arrays, primary_audit = load_numeric_archive(primary_source)
    crosscheck_arrays, crosscheck_audit = load_numeric_archive(crosscheck_source)

    for arrays in (primary_arrays, crosscheck_arrays):
        if not np.array_equal(
            arrays["parent_active_mo_indices"], EXPECTED_ACTIVE_INDICES
        ):
            raise ValueError("numeric archive has the wrong parent active indices")

    primary_mos = np.asarray(primary_arrays["parent_to_rhf_mo_coeff"])
    crosscheck_mos = np.asarray(crosscheck_arrays["parent_to_rhf_mo_coeff"])
    primary_projector = primary_mos[:, 11:19] @ primary_mos[:, 11:19].T
    crosscheck_projector = crosscheck_mos[:, 11:19] @ crosscheck_mos[:, 11:19].T

    p_results = primary["results"]
    x_results = crosscheck["results"]
    p_desc = p_results["correlation_descriptors"]
    x_desc = x_results["correlation_descriptors"]
    p_controls = p_results["classical_controls"]
    x_controls = x_results["classical_controls"]
    comparisons = {
        "casci_energy_difference_hartree": scalar_difference(
            p_results["casci_energy_hartree"], x_results["casci_energy_hartree"]
        ),
        "cisd_energy_difference_hartree": scalar_difference(
            p_controls["cisd_energy_hartree"], x_controls["cisd_energy_hartree"]
        ),
        "ccsd_energy_difference_hartree": scalar_difference(
            p_controls["ccsd_energy_hartree"], x_controls["ccsd_energy_hartree"]
        ),
        "renyi_h0p25_difference_nats": scalar_difference(
            p_desc["renyi_h0p25_nats"], x_desc["renyi_h0p25_nats"]
        ),
        "cumulant_C2_difference": scalar_difference(
            p_desc["cumulant_C2"], x_desc["cumulant_C2"]
        ),
        "largest_determinant_probability_difference": scalar_difference(
            p_desc["largest_determinant_probability"],
            x_desc["largest_determinant_probability"],
        ),
        "n92_equal": bool(p_desc["n92"] == x_desc["n92"]),
        "active_projector_frobenius_difference": float(
            np.linalg.norm(primary_projector - crosscheck_projector)
        ),
        "one_body_integral_frobenius_difference_hartree": float(
            np.linalg.norm(
                primary_arrays["one_body_integrals"]
                - crosscheck_arrays["one_body_integrals"]
            )
        ),
        "two_body_integral_frobenius_difference_hartree": float(
            np.linalg.norm(
                primary_arrays["two_body_integrals"]
                - crosscheck_arrays["two_body_integrals"]
            )
        ),
        "core_constant_difference_hartree": float(
            np.linalg.norm(
                primary_arrays["core_constant"]
                - crosscheck_arrays["core_constant"]
            )
        ),
    }
    gates = {
        "primary_and_crosscheck_reports_accepted": True,
        "numeric_archives_pickle_free_and_finite": True,
        "parent_active_indices_identical": True,
        "casci_energy_cross_version": (
            comparisons["casci_energy_difference_hartree"] <= 1.0e-10
        ),
        "cisd_energy_cross_version": (
            comparisons["cisd_energy_difference_hartree"] <= 1.0e-9
        ),
        "ccsd_energy_cross_version": (
            comparisons["ccsd_energy_difference_hartree"] <= 1.0e-9
        ),
        "renyi_h0p25_cross_version": (
            comparisons["renyi_h0p25_difference_nats"] <= 1.0e-8
        ),
        "cumulant_C2_cross_version": (
            comparisons["cumulant_C2_difference"] <= 1.0e-8
        ),
        "largest_determinant_cross_version": (
            comparisons["largest_determinant_probability_difference"] <= 1.0e-8
        ),
        "n92_cross_version": comparisons["n92_equal"],
        "active_projector_cross_version": (
            comparisons["active_projector_frobenius_difference"] <= 1.0e-7
        ),
        "active_one_body_cross_version": (
            comparisons["one_body_integral_frobenius_difference_hartree"] <= 1.0e-7
        ),
        "active_two_body_cross_version": (
            comparisons["two_body_integral_frobenius_difference_hartree"] <= 1.0e-7
        ),
        "active_core_constant_cross_version": (
            comparisons["core_constant_difference_hartree"] <= 1.0e-7
        ),
    }
    passed = all(gates.values())
    certificate = {
        "schema": "benchmark-qc.fe2s2-cross-version-certificate.v1",
        "status": "accepted" if passed else "rejected",
        "dataset_id": EXPECTED_DATASET_ID,
        "claim_boundary": (
            "Independent PySCF 2.4/2.11 agreement for the reconstructed "
            "nested CAS(8e,8o) extension; not a historical paper row"
        ),
        "inputs": {
            "primary_report": {
                "path": str(primary_path),
                "sha256": sha256_file(primary_path),
                "source_integrals_sha256": sha256_file(primary_source),
                "hamiltonian_sha256": sha256_file(primary_hamiltonian),
            },
            "crosscheck_report": {
                "path": str(crosscheck_path),
                "sha256": sha256_file(crosscheck_path),
                "source_integrals_sha256": sha256_file(crosscheck_source),
                "hamiltonian_sha256": sha256_file(crosscheck_hamiltonian),
            },
        },
        "comparisons": comparisons,
        "gates": gates,
        "numeric_archive_audit": {
            "primary": primary_audit,
            "crosscheck": crosscheck_audit,
        },
        "runtime": {
            "hostname": hostname,
            "username": os.environ.get("USER"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"status": certificate["status"], **comparisons}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
