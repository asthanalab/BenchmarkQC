"""Access and acceptance helpers for the Fe2S2 benchmark.

The Fe2S2 workflow is intentionally acceptance-first.  Candidate calculations
live below ``datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/candidates`` and are not exposed by the
package or root catalog.  Only the promotion script may place a validated
variant below ``accepted`` and add its checksum-pinned entry to
``datasets/catalog.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hamiltonian_test import load_hamiltonian_npz
from .integral_dataset import load_spatial_integral_archive
from .paths import BENCHMARKQC_ROOT


DATASET_ROOT = BENCHMARKQC_ROOT / "source" / "molecules" / "Fe2S2" / "cas6e6o_chan30e20o"
PARENT_FCIDUMP_PATH = (
    DATASET_ROOT / "parent" / "chan_fe2s2_cas30e20o.FCIDUMP"
)
PARENT_FCIDUMP_SHA256 = (
    "95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9"
)
HISTORICAL_SOURCE_PATH = DATASET_ROOT / "provenance" / "vi66cycle_historical_source.py"
HISTORICAL_SOURCE_SHA256 = (
    "388897d39451399f506b8d293c6e2465db9882deec61aa5da1e70075dec612b7"
)
PUBLISHED_HISTORICAL_ENERGIES_HARTREE = {
    "historical_cas4e4o": -116.154749,
    "historical_cas6e6o": -116.156259,
    "historical_cas8e6o": -116.158572,
}
PUBLISHED_CAS6_ENERGY_HARTREE = PUBLISHED_HISTORICAL_ENERGIES_HARTREE[
    "historical_cas6e6o"
]
PUBLISHED_PARENT_DMRG_ENERGY_HARTREE = -116.6056091
PUBLISHED_VALUE_PRECISION_HARTREE = 1.0e-6

DATASET_IDS = frozenset(
    {
        "fe2s2_chan30e20o_historical_rhf_cas6e6o",
        "fe2s2_chan30e20o_historical_rhf_cas4e4o",
        "fe2s2_chan30e20o_historical_rhf_cas8e6o",
        "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o",
        "fe2s2_chan30e20o_asf_dmrg_cas6e6o",
    }
)
VARIANT_DATASET_IDS = {
    "historical_cas4e4o": "fe2s2_chan30e20o_historical_rhf_cas4e4o",
    "historical_cas6e6o": "fe2s2_chan30e20o_historical_rhf_cas6e6o",
    "historical_cas8e6o": "fe2s2_chan30e20o_historical_rhf_cas8e6o",
    "nested_cas8e8o": "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o",
    "asf_dmrg_cas6e6o": "fe2s2_chan30e20o_asf_dmrg_cas6e6o",
}
ACCEPTED_VARIANTS = {
    variant: DATASET_ROOT / "accepted" / variant for variant in VARIANT_DATASET_IDS
}


class CandidateNotAcceptedError(RuntimeError):
    """Raised when a caller tries to publish or load an unaccepted candidate."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_boolean_gates(report: Mapping[str, Any]) -> None:
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise CandidateNotAcceptedError("candidate report has no acceptance gates")
    non_boolean = sorted(key for key, value in gates.items() if not isinstance(value, bool))
    if non_boolean:
        raise CandidateNotAcceptedError(
            f"candidate gates are not Boolean: {non_boolean}"
        )
    failed = sorted(key for key, value in gates.items() if not value)
    if failed:
        raise CandidateNotAcceptedError(f"candidate failed gates: {failed}")


def require_accepted_report(report: Mapping[str, Any]) -> None:
    """Validate the non-negotiable publication fields in a candidate report.

    File checksums are validated by :func:`require_accepted_artifacts`, because
    paths are resolved relative to the report after a remote result is copied
    back to the repository.
    """

    if report.get("schema") != "benchmark-qc.fe2s2-candidate.v1":
        raise CandidateNotAcceptedError("unrecognized Fe2S2 candidate schema")
    if report.get("dataset_id") not in DATASET_IDS:
        raise CandidateNotAcceptedError("unrecognized Fe2S2 dataset identifier")
    variant = report.get("variant")
    if variant not in VARIANT_DATASET_IDS or (
        report.get("dataset_id") != VARIANT_DATASET_IDS[variant]
    ):
        raise CandidateNotAcceptedError("Fe2S2 variant and dataset identifier disagree")
    if report.get("status") != "accepted":
        raise CandidateNotAcceptedError(
            f"candidate status is {report.get('status')!r}, not 'accepted'"
        )
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("execution_site") not in {
        "desktop",
        "talon",
    }:
        raise CandidateNotAcceptedError(
            "candidate must record execution_site as desktop or talon"
        )
    _require_boolean_gates(report)

    results = report.get("results")
    if not isinstance(results, Mapping):
        raise CandidateNotAcceptedError("candidate report has no results object")
    energy = float(results.get("casci_energy_hartree", float("nan")))
    if variant == "nested_cas8e8o":
        required_nested_gates = {
            "active_space_fci_converged",
            "active_space_physical_residual",
            "active_space_singlet",
            "active_space_spin_eigenvector",
            "active_space_rdm_energy",
            "historical_cas6_control_energy",
            "historical_cas6_control_state",
            "historical_default_rhf_remains_nonconverged",
            "historical_rhf_energy_fingerprint",
            "historical_rhf_gradient_fingerprint",
            "historical_rhf_mo_fingerprint_policy",
            "nested_default_partition",
            "strictly_contains_historical_cas6_span",
            "variational_nesting_energy",
        }
        if not required_nested_gates.issubset(report["gates"]):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 report is missing required acceptance gates"
            )
        target = report.get("target", {})
        method = report.get("method", {})
        acceptance = report.get("acceptance", {})
        solver = results.get("solver", {})
        cas6_control = results.get("historical_cas6_same_process_control", {})
        if (
            target.get("active_electrons") != 8
            or target.get("active_spatial_orbitals") != 8
            or target.get("qubits") != 16
            or target.get("spin_2S") != 0
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 report has the wrong CAS(8e,8o) target"
            )
        if (
            method.get("historical_paper_space") is not False
            or method.get("active_space_finder_used") is not False
            or method.get("stationary_rhf_claim") is not False
            or method.get("active_indices_in_rhf_frame_zero_based")
            != list(range(11, 19))
            or method.get("nested_parent_space", {}).get(
                "historical_cas6_indices_zero_based"
            )
            != list(range(12, 18))
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 method provenance or orbital span is invalid"
            )
        if (
            acceptance.get("mode") != "reconstructed_nested_extension"
            or acceptance.get("published_energy_target_used") is not False
            or acceptance.get("runtime_role")
            not in {"primary_pyscf2p4", "crosscheck_pyscf2p11"}
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 acceptance mode is invalid"
            )
        if (
            not math.isfinite(energy)
            or float(solver.get("physical_hamiltonian_residual_norm_hartree", math.inf))
            > 1.0e-9
            or float(solver.get("spin_eigenvector_residual_norm", math.inf)) > 1.0e-8
            or not bool(solver.get("converged"))
            or not math.isfinite(
                float(cas6_control.get("casci_energy_hartree", math.nan))
            )
            or energy > float(cas6_control["casci_energy_hartree"]) + 1.0e-10
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 physical or variational certificate failed"
            )
        return

    published_energy = (
        PUBLISHED_HISTORICAL_ENERGIES_HARTREE.get(variant)
        if variant != "asf_dmrg_cas6e6o"
        else PUBLISHED_CAS6_ENERGY_HARTREE
    )
    tolerance = float(
        report.get("acceptance", {}).get(
            "energy_tolerance_hartree",
            report.get("acceptance", {}).get("cas6_energy_tolerance_hartree", 0.0),
        )
    )
    if (
        not math.isfinite(energy)
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
        or tolerance > PUBLISHED_VALUE_PRECISION_HARTREE
        or published_energy is None
        or abs(energy - published_energy) > tolerance
    ):
        raise CandidateNotAcceptedError(
            "candidate active-space energy does not reproduce the published value"
        )


def require_accepted_artifacts(
    report: Mapping[str, Any], *, report_directory: str | Path
) -> dict[str, Path]:
    """Validate all checksum-pinned candidate artifacts and return their paths."""

    require_accepted_report(report)
    root = Path(report_directory).resolve()
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CandidateNotAcceptedError("candidate report has no artifact manifest")

    validated: dict[str, Path] = {}
    for key in ("source_integrals", "hamiltonian"):
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise CandidateNotAcceptedError(f"missing artifact record {key!r}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise CandidateNotAcceptedError(
                f"artifact {key!r} must use a safe report-relative path"
            )
        path = (root / relative).resolve()
        if root not in path.parents:
            raise CandidateNotAcceptedError(f"artifact {key!r} leaves report directory")
        if not path.is_file():
            raise CandidateNotAcceptedError(f"artifact {key!r} is missing: {path}")
        expected = str(record.get("sha256", ""))
        actual = sha256_file(path)
        if actual != expected:
            raise CandidateNotAcceptedError(
                f"artifact {key!r} checksum mismatch: expected {expected}, got {actual}"
            )
        validated[key] = path
    return validated


def require_nested_cross_version_validation(
    report: Mapping[str, Any], *, report_directory: str | Path
) -> None:
    """Validate the promoted CAS(8e,8o) cross-version certificate and inputs."""

    if report.get("variant") != "nested_cas8e8o":
        raise CandidateNotAcceptedError(
            "cross-version validation is defined only for nested_cas8e8o"
        )
    root = Path(report_directory).resolve()
    validation = root / "validation"
    crosscheck_path = validation / "crosscheck_candidate_report.json"
    certificate_path = validation / "cross_version_certificate.json"
    if not crosscheck_path.is_file() or not certificate_path.is_file():
        raise CandidateNotAcceptedError(
            "nested Fe2S2 cross-version report or certificate is missing"
        )

    try:
        crosscheck = json.loads(crosscheck_path.read_text(encoding="utf-8"))
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise CandidateNotAcceptedError(
            "nested Fe2S2 cross-version validation is unreadable"
        ) from error
    require_accepted_artifacts(crosscheck, report_directory=validation)
    if (
        crosscheck.get("variant") != "nested_cas8e8o"
        or crosscheck.get("acceptance", {}).get("runtime_role")
        != "crosscheck_pyscf2p11"
    ):
        raise CandidateNotAcceptedError(
            "nested Fe2S2 cross-check has the wrong role or variant"
        )

    gates = certificate.get("gates")
    primary_input = certificate.get("inputs", {}).get("primary_report", {})
    crosscheck_input = certificate.get("inputs", {}).get(
        "crosscheck_report", {}
    )
    if (
        certificate.get("schema")
        != "benchmark-qc.fe2s2-cross-version-certificate.v1"
        or certificate.get("status") != "accepted"
        or certificate.get("dataset_id") != report.get("dataset_id")
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or primary_input.get("path") != "../acceptance_report.json"
        or crosscheck_input.get("path") != "crosscheck_candidate_report.json"
        or primary_input.get("sha256")
        != sha256_file(root / "acceptance_report.json")
        or crosscheck_input.get("sha256") != sha256_file(crosscheck_path)
    ):
        raise CandidateNotAcceptedError(
            "nested Fe2S2 cross-version certificate is invalid"
        )

    metadata_path = DATASET_ROOT / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_certificate_sha256 = metadata["accepted_variants"][
            "nested_cas8e8o"
        ]["cross_version_validation"]["certificate_sha256"]
    except (KeyError, json.JSONDecodeError, OSError, TypeError) as error:
        raise CandidateNotAcceptedError(
            "nested Fe2S2 metadata lacks the promoted certificate checksum"
        ) from error
    if sha256_file(certificate_path) != expected_certificate_sha256:
        raise CandidateNotAcceptedError(
            "nested Fe2S2 certificate checksum disagrees with metadata"
        )

    primary_artifacts = report.get("artifacts", {})
    crosscheck_artifacts = crosscheck.get("artifacts", {})
    for input_record, artifacts in (
        (primary_input, primary_artifacts),
        (crosscheck_input, crosscheck_artifacts),
    ):
        if (
            input_record.get("source_integrals_sha256")
            != artifacts.get("source_integrals", {}).get("sha256")
            or input_record.get("hamiltonian_sha256")
            != artifacts.get("hamiltonian", {}).get("sha256")
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 certificate artifact checksums disagree"
            )

    primary_results = report["results"]
    crosscheck_results = crosscheck["results"]
    primary_controls = primary_results["classical_controls"]
    crosscheck_controls = crosscheck_results["classical_controls"]
    primary_descriptors = primary_results["correlation_descriptors"]
    crosscheck_descriptors = crosscheck_results["correlation_descriptors"]
    recomputed = {
        "casci_energy_difference_hartree": abs(
            float(primary_results["casci_energy_hartree"])
            - float(crosscheck_results["casci_energy_hartree"])
        ),
        "cisd_energy_difference_hartree": abs(
            float(primary_controls["cisd_energy_hartree"])
            - float(crosscheck_controls["cisd_energy_hartree"])
        ),
        "ccsd_energy_difference_hartree": abs(
            float(primary_controls["ccsd_energy_hartree"])
            - float(crosscheck_controls["ccsd_energy_hartree"])
        ),
        "renyi_h0p25_difference_nats": abs(
            float(primary_descriptors["renyi_h0p25_nats"])
            - float(crosscheck_descriptors["renyi_h0p25_nats"])
        ),
        "cumulant_C2_difference": abs(
            float(primary_descriptors["cumulant_C2"])
            - float(crosscheck_descriptors["cumulant_C2"])
        ),
        "largest_determinant_probability_difference": abs(
            float(primary_descriptors["largest_determinant_probability"])
            - float(crosscheck_descriptors["largest_determinant_probability"])
        ),
        "n92_equal": bool(
            primary_descriptors["n92"] == crosscheck_descriptors["n92"]
        ),
    }

    numeric_archives: list[dict[str, np.ndarray]] = []
    for source_path in (
        root / "inputs" / "source_integrals.npz",
        validation / "crosscheck_source_integrals.npz",
    ):
        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(source_path, allow_pickle=False) as archive:
                for name in archive.files:
                    value = np.asarray(archive[name])
                    if value.dtype.kind == "O" or (
                        value.dtype.kind in "fc" and not np.all(np.isfinite(value))
                    ):
                        raise CandidateNotAcceptedError(
                            "nested Fe2S2 numeric archive is non-finite or pickled"
                        )
                    arrays[name] = value
        except (OSError, ValueError) as error:
            raise CandidateNotAcceptedError(
                "nested Fe2S2 numeric archive cannot be validated"
            ) from error
        if not np.array_equal(
            arrays.get("parent_active_mo_indices"),
            np.arange(11, 19, dtype=np.int64),
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 numeric archive has the wrong active indices"
            )
        numeric_archives.append(arrays)

    primary_arrays, crosscheck_arrays = numeric_archives
    primary_mos = np.asarray(primary_arrays["parent_to_rhf_mo_coeff"])
    crosscheck_mos = np.asarray(crosscheck_arrays["parent_to_rhf_mo_coeff"])
    primary_projector = primary_mos[:, 11:19] @ primary_mos[:, 11:19].T
    crosscheck_projector = crosscheck_mos[:, 11:19] @ crosscheck_mos[:, 11:19].T
    recomputed.update(
        {
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
    )

    thresholds = {
        "casci_energy_difference_hartree": 1.0e-10,
        "cisd_energy_difference_hartree": 1.0e-9,
        "ccsd_energy_difference_hartree": 1.0e-9,
        "renyi_h0p25_difference_nats": 1.0e-8,
        "cumulant_C2_difference": 1.0e-8,
        "largest_determinant_probability_difference": 1.0e-8,
        "active_projector_frobenius_difference": 1.0e-7,
        "one_body_integral_frobenius_difference_hartree": 1.0e-7,
        "two_body_integral_frobenius_difference_hartree": 1.0e-7,
        "core_constant_difference_hartree": 1.0e-7,
    }
    recorded = certificate.get("comparisons", {})
    for key, limit in thresholds.items():
        value = recomputed[key]
        if (
            not math.isfinite(value)
            or value > limit
            or not math.isclose(
                value,
                float(recorded.get(key, math.nan)),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise CandidateNotAcceptedError(
                "nested Fe2S2 cross-version comparison failed"
            )
    if recomputed["n92_equal"] is not True or recorded.get("n92_equal") is not True:
        raise CandidateNotAcceptedError(
            "nested Fe2S2 N92 cross-version comparison failed"
        )


def accepted_variant_available(variant: str) -> bool:
    """Return whether a promoted, checksum-pinned variant is present."""

    root = ACCEPTED_VARIANTS.get(variant)
    if root is None:
        return False
    report_path = root / "acceptance_report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("variant") != variant:
            return False
        require_accepted_artifacts(report, report_directory=root)
        if variant == "nested_cas8e8o":
            require_nested_cross_version_validation(
                report, report_directory=root
            )
    except (CandidateNotAcceptedError, json.JSONDecodeError, OSError, ValueError):
        return False
    return True


def _accepted_root(variant: str) -> Path:
    try:
        root = ACCEPTED_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown Fe2S2 variant {variant!r}; choose {sorted(ACCEPTED_VARIANTS)}"
        ) from error
    if not accepted_variant_available(variant):
        raise CandidateNotAcceptedError(
            f"Fe2S2 variant {variant!r} has not passed promotion; "
            "run the documented remote calculation and acceptance workflow first"
        )
    return root


def load_hamiltonian(variant: str = "historical_cas6e6o"):
    """Load one promoted historical PennyLane Hamiltonian archive."""

    return load_hamiltonian_npz(str(_accepted_root(variant) / "Fe2S2_H.npz"))


def load_source_integrals(variant: str = "historical_cas6e6o"):
    """Load one promoted pickle-free spatial-integral archive."""

    return load_spatial_integral_archive(
        _accepted_root(variant) / "inputs" / "source_integrals.npz"
    )
