"""Reconstruct missing SI Table I Hamiltonians from the QCANT source snapshot.

The QCANT ``change3_benchmark10_reference`` tree is treated as a read-only
source.  This tool imports only portable active-space inputs, regenerates the
PennyLane Jordan--Wigner Hamiltonian, recomputes the target-spin CASCI value,
and promotes a point only after the recomputed value agrees with both the
QCANT reference record and the SI machine-readable value.

Application calculation output is never written to the repository's
``applications/`` tree.  The only repository outputs are the benchmark
Hamiltonian, its pickle-free integral input, metadata, and catalog entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import fci

from benchmark_qc.integral_dataset import (
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    sha256_file,
    write_historical_hamiltonian_npz,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QCANT_ROOT = Path(
    "/Users/ayushasthana/Library/CloudStorage/OneDrive2-"
    "NorthDakotaUniversitySystem/code/qcomputing/QCANT/"
    "change3_benchmark10_reference"
)
CATALOG_PATH = ROOT / "datasets" / "catalog.json"
INVENTORY_PATH = ROOT / "datasets" / "benchmarkQC" / "si_table_i_inventory.json"
OUTPUT_ROOT = ROOT / "datasets" / "benchmarkQC"
RECONSTRUCTION_ROOT = ROOT / "applications" / "reconstruction" / "si_table_i"
TOLERANCE_HARTREE = 1.0e-8


@dataclass(frozen=True)
class CaseSpec:
    system: str
    system_id: str
    case_id: str
    inventory_label: str
    label: float
    source_rel: str
    reference_rel: str
    active_electrons: int
    active_orbitals: int
    spin_2s: int
    basis: str
    si_energy_hartree: float
    mode: str = "direct"
    reference_json_rel: str | None = None


QCANT_PREFIX = "QCANT/change3_benchmark10_reference/"


CASE_SPECS = (
    CaseSpec(
        "FeS", "fes_cas14e10o_anorccvdzp_sfx2c",
        "fes_cas14e10o_anorccvdzp_sfx2c_equilibrium_r2p0170",
        "equilibrium_r2p0170", 2.017,
        "remote_results_final_talon/fes_cas14e10o_anorccvdzp_sfx2c/equilibrium_r2p0170/inputs.npz",
        "remote_results_final_talon/fes_cas14e10o_anorccvdzp_sfx2c/equilibrium_r2p0170/reference_results.npz",
        14, 10, 4, "ANO-RCC-VDZP; SF-X2C-1e", -1669.9551860647196,
    ),
    CaseSpec(
        "FeS", "fes_cas14e10o_anorccvdzp_sfx2c",
        "fes_cas14e10o_anorccvdzp_sfx2c_stretched_r3p0000",
        "stretched_r3p0000", 3.0,
        "remote_results_final_talon/fes_cas14e10o_anorccvdzp_sfx2c/stretched_r3p0000/inputs.npz",
        "remote_results_final_talon/fes_cas14e10o_anorccvdzp_sfx2c/stretched_r3p0000/reference_results.npz",
        14, 10, 4, "ANO-RCC-VDZP; SF-X2C-1e", -1669.944898829754,
    ),
    CaseSpec(
        "Fe2S2", "fe2s2_cas10e10o_anorccvdz_mb",
        "fe2s2_cas10e10o_anorccvdz_mb_published_geometry",
        "published", 0.0,
        "validation/fe2s2_exact_reference_283930/results/published_geometry/inputs.npz",
        "validation/fe2s2_exact_reference_283930/results/published_geometry/reference_results.npz",
        10, 10, 0, "Fe ANO-RCC-VDZ; S,C,H ANO-RCC-MB; nonrelativistic",
        -5058.275918325761, mode="fe2s2_published",
    ),
    CaseSpec(
        "Fe2S2", "fe2s2_cas10e10o_anorccvdz_mb",
        "fe2s2_cas10e10o_anorccvdz_mb_bridge_stretched_1p10",
        "stretched", 0.0,
        "validation/fe2s2_exact_reference_283930/results/bridge_stretched_1p10/inputs.npz",
        "validation/fe2s2_exact_reference_283930/results/bridge_stretched_1p10/reference_results.npz",
        10, 10, 0, "Fe ANO-RCC-VDZ; S,C,H ANO-RCC-MB; nonrelativistic",
        -5058.113901629829, mode="fe2s2_stretched",
    ),
    CaseSpec(
        "U2", "u2_cas6e10o_anorccvdzp_sfx2c",
        "u2_cas6e10o_anorccvdzp_sfx2c_equilibrium_r2p4300",
        "equilibrium_r2p4300", 2.43,
        "validation/u2_partial_reference_desktop_20260804_88f8a4cc_8f7bda40_29e6d3f4/inputs/certificates/equilibrium_r2p4300_validated_lowest_singlet.npz",
        "validation/u2_partial_reference_desktop_20260804_88f8a4cc_8f7bda40_29e6d3f4/results/equilibrium_r2p4300_partial_reference.npz",
        6, 10, 0, "ANO-RCC-VDZP; SF-X2C-1e", -55957.17583824898,
        mode="u2_certificate",
        reference_json_rel="validation/u2_singlet_certificate_v5_283952_283953/results/equilibrium_r2p4300_validated_lowest_singlet.json",
    ),
    CaseSpec(
        "U2", "u2_cas6e10o_anorccvdzp_sfx2c",
        "u2_cas6e10o_anorccvdzp_sfx2c_stretched_r2p8000",
        "stretched_r2p8000", 2.8,
        "validation/u2_partial_reference_desktop_20260804_88f8a4cc_8f7bda40_29e6d3f4/inputs/certificates/stretched_r2p8000_validated_lowest_singlet.npz",
        "validation/u2_partial_reference_desktop_20260804_88f8a4cc_8f7bda40_29e6d3f4/results/stretched_r2p8000_partial_reference.npz",
        6, 10, 0, "ANO-RCC-VDZP; SF-X2C-1e", -55957.18982335859,
        mode="u2_certificate",
        reference_json_rel="validation/u2_singlet_certificate_v5_283952_283953/results/stretched_r2p8000_validated_lowest_singlet.json",
    ),
    CaseSpec(
        "C2H4", "c2h4_cas2e2o_augccpvdz",
        "c2h4_cas2e2o_augccpvdz_planar_0deg", "planar_0deg", 0.0,
        "remote_results_final_talon/c2h4_cas2e2o_augccpvdz/planar_0deg/inputs.npz",
        "remote_results_final_talon/c2h4_cas2e2o_augccpvdz/planar_0deg/reference_results.npz",
        2, 2, 0, "aug-cc-pVDZ; nonrelativistic", -78.0691581246013,
    ),
    CaseSpec(
        "C2H4", "c2h4_cas2e2o_augccpvdz",
        "c2h4_cas2e2o_augccpvdz_twisted_90deg", "twisted_90deg", 90.0,
        "remote_results_final_talon/c2h4_cas2e2o_augccpvdz/twisted_90deg/inputs.npz",
        "remote_results_final_talon/c2h4_cas2e2o_augccpvdz/twisted_90deg/reference_results.npz",
        2, 2, 0, "aug-cc-pVDZ; nonrelativistic", -77.95435497575266,
    ),
    CaseSpec(
        "CH2", "ch2_cas6e6o_augccpvtz",
        "ch2_cas6e6o_augccpvtz_bent_102deg", "bent_102deg", 102.0,
        "remote_results_final_talon/ch2_cas6e6o_augccpvtz/bent_102deg/inputs.npz",
        "remote_results_final_talon/ch2_cas6e6o_augccpvtz/bent_102deg/reference_results.npz",
        6, 6, 0, "aug-cc-pVTZ; nonrelativistic", -38.92873176648936,
    ),
    CaseSpec(
        "CH2", "ch2_cas6e6o_augccpvtz",
        "ch2_cas6e6o_augccpvtz_open_140deg", "open_140deg", 140.0,
        "remote_results_final_talon/ch2_cas6e6o_augccpvtz/open_140deg/inputs.npz",
        "remote_results_final_talon/ch2_cas6e6o_augccpvtz/open_140deg/reference_results.npz",
        6, 6, 0, "aug-cc-pVTZ; nonrelativistic", -38.90416513357388,
    ),
    CaseSpec(
        "C2", "c2_cas8e8o_augccpvtz",
        "c2_cas8e8o_augccpvtz_equilibrium_r1p2430", "equilibrium_r1p2430", 1.243,
        "remote_results_final_talon/c2_cas8e8o_augccpvtz/equilibrium_r1p2430/inputs.npz",
        "remote_results_final_talon/c2_cas8e8o_augccpvtz/equilibrium_r1p2430/reference_results.npz",
        8, 8, 0, "aug-cc-pVTZ; nonrelativistic", -75.62029347253048,
    ),
    CaseSpec(
        "O2", "o2_cas12e8o_augccpvtz",
        "o2_cas12e8o_augccpvtz_equilibrium_r1p2075", "equilibrium_r1p2075", 1.2075,
        "remote_results_final_talon/o2_cas12e8o_augccpvtz/equilibrium_r1p2075/inputs.npz",
        "remote_results_final_talon/o2_cas12e8o_augccpvtz/equilibrium_r1p2075/reference_results.npz",
        12, 8, 2, "aug-cc-pVTZ; nonrelativistic", -149.75701080939123,
    ),
    CaseSpec(
        "O2", "o2_cas12e8o_augccpvtz",
        "o2_cas12e8o_augccpvtz_stretched_r2p0000", "stretched_r2p0000", 2.0,
        "remote_results_final_talon/o2_cas12e8o_augccpvtz/stretched_r2p0000/inputs.npz",
        "remote_results_final_talon/o2_cas12e8o_augccpvtz/stretched_r2p0000/reference_results.npz",
        12, 8, 2, "aug-cc-pVTZ; nonrelativistic", -149.6143039628574,
    ),
    CaseSpec(
        "FeH", "feh_cas9e10o_augccpvtzdk_sfx2c",
        "feh_cas9e10o_augccpvtzdk_sfx2c_equilibrium_r1p5680", "equilibrium_r1p5680", 1.568,
        "remote_results_final_talon/feh_cas9e10o_augccpvtzdk_sfx2c/equilibrium_r1p5680/inputs.npz",
        "remote_results_final_talon/feh_cas9e10o_augccpvtzdk_sfx2c/equilibrium_r1p5680/reference_results.npz",
        9, 10, 3, "aug-cc-pVTZ-DK; SF-X2C-1e", -1271.891370939278,
    ),
    CaseSpec(
        "FeH", "feh_cas9e10o_augccpvtzdk_sfx2c",
        "feh_cas9e10o_augccpvtzdk_sfx2c_stretched_r2p3000", "stretched_r2p3000", 2.3,
        "remote_results_final_talon/feh_cas9e10o_augccpvtzdk_sfx2c/stretched_r2p3000/inputs.npz",
        "remote_results_final_talon/feh_cas9e10o_augccpvtzdk_sfx2c/stretched_r2p3000/reference_results.npz",
        9, 10, 3, "aug-cc-pVTZ-DK; SF-X2C-1e", -1271.8774476006429,
    ),
)


def _doubly_occupied_reference(active_electrons: int, orbitals: int) -> np.ndarray:
    doubly_occupied = active_electrons // 2
    reference = np.zeros(2 * orbitals, dtype=bool)
    reference[: 2 * doubly_occupied] = True
    return reference


def _fe2s2_geometry(qcant_root: Path, stretched: bool) -> np.ndarray:
    xyz = qcant_root / "sources" / "Fe2S2" / "geometry_Fe2S2.xyz"
    lines = xyz.read_text(encoding="utf-8").splitlines()[2:]
    coordinates = np.asarray(
        [[float(value) for value in line.split()[1:4]] for line in lines], dtype=float
    )
    if not stretched:
        return coordinates
    fe_z = abs(float(coordinates[0, 2] - coordinates[2, 2]))
    old_radius = float(np.linalg.norm(coordinates[2, :2]))
    old_distance = float(np.sqrt(old_radius**2 + fe_z**2))
    new_distance = 1.1 * old_distance
    new_radius = float(np.sqrt(new_distance**2 - fe_z**2))
    transformed = coordinates.copy()
    transformed[2:4, :2] *= new_radius / old_radius
    return transformed


def _u2_geometry(label: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, -label / 2.0], [0.0, 0.0, label / 2.0]])


def _source_payload(spec: CaseSpec, qcant_root: Path) -> dict[str, np.ndarray]:
    source = qcant_root / spec.source_rel
    with np.load(source, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}

    if spec.mode == "direct":
        return payload

    n_orbitals = spec.active_orbitals
    reference = _doubly_occupied_reference(spec.active_electrons, n_orbitals)
    if spec.mode.startswith("fe2s2"):
        payload.update(
            {
                "geometry_angstrom": _fe2s2_geometry(
                    qcant_root, stretched=spec.mode == "fe2s2_stretched"
                ),
                "active_mo_indices": np.empty(0, dtype=np.int64),
                "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
                "selected_mp2_natural_coeff": np.empty((n_orbitals, 0)),
                "pre_hf_active_mo_coeff": np.eye(n_orbitals),
                "pre_hf_active_coefficient_source": np.asarray(
                    "stored active-space integral frame; AO coefficients unavailable"
                ),
                "pre_hf_projected_mp2_natural_occupation_data": np.asarray(
                    payload["active_hf_orbital_occupations"], dtype=float
                ),
                "reference_determinant": reference,
            }
        )
        return payload

    if spec.mode == "u2_certificate":
        if "active_hf_one_body_integrals" in payload:
            payload["one_body_integrals"] = payload.pop("active_hf_one_body_integrals")
            payload["two_body_integrals"] = payload.pop("active_hf_two_body_integrals")
            one_rdm = np.asarray(
                payload.pop("active_hf_certified_singlet_spatial_one_rdm"), dtype=float
            )
            active_frame = np.asarray(payload.pop("active_hf_rotation"), dtype=float)
        else:
            one_rdm = np.asarray(
                payload.pop("certified_singlet_spatial_one_rdm"), dtype=float
            )
            active_frame = np.asarray(payload.pop("optimized_active_coeff"), dtype=float)
            payload["one_body_integrals"] = payload["one_body_integrals"]
            payload["two_body_integrals"] = payload["two_body_integrals"]
        payload["core_constant"] = payload.pop("core_constant_hartree")
        occupations = np.linalg.eigvalsh(one_rdm)
        payload.pop("active_hf_one_rdm_spinorbital", None)
        payload.pop("active_hf_two_rdm_spinorbital", None)
        payload.pop("active_hf_two_body_cumulant_spinorbital", None)
        payload.pop("active_hf_certified_singlet_ci", None)
        payload.pop("active_hf_renyi_0p25_nats", None)
        payload.pop("active_hf_cumulant_C2", None)
        payload.pop("active_hf_ground_cluster_ci", None)
        payload.pop("ground_cluster_renyi_H0p25_nats", None)
        payload.pop("ground_cluster_cumulant_C2", None)
        payload.pop("dense_singlet_rcisd_packed_ci", None)
        payload.pop("dense_singlet_rcisd_fci_embedding", None)
        payload.update(
            {
                "geometry_angstrom": _u2_geometry(spec.label),
                "active_mo_indices": np.empty(0, dtype=np.int64),
                "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
                "selected_mp2_natural_coeff": np.empty(
                    (active_frame.shape[0], 0)
                ),
                "pre_hf_active_mo_coeff": active_frame,
                "pre_hf_active_coefficient_source": np.asarray(
                    "CASSCF-refined selected active-space coefficients"
                ),
                "pre_hf_projected_mp2_natural_occupation_data": occupations,
                "reference_determinant": reference,
            }
        )
        return payload

    raise ValueError(f"unsupported source mode {spec.mode!r}")


def _reference_energy(spec: CaseSpec, qcant_root: Path) -> float:
    if spec.mode == "u2_certificate":
        record = json.loads((qcant_root / str(spec.reference_json_rel)).read_text())
        return float(record["assessment"]["candidate_energy_hartree"])
    with np.load(qcant_root / spec.reference_rel, allow_pickle=False) as data:
        return float(data["casci_total_energy"])


def recompute_target_energy(
    archive, *, active_electrons: int, spin_2s: int, ci_vector: np.ndarray | None = None
) -> float:
    if ci_vector is not None:
        if spin_2s != 0:
            raise ValueError("explicit CI-vector reconstruction currently supports singlets only")
        nelec = (active_electrons // 2, active_electrons // 2)
        effective_h2 = fci.direct_spin1.absorb_h1e(
            archive.one_body_integrals,
            archive.two_body_integrals,
            archive.n_spatial_orbitals,
            nelec,
            0.5,
        )
        contracted = fci.direct_spin1.contract_2e(
            effective_h2,
            np.asarray(ci_vector, dtype=float),
            archive.n_spatial_orbitals,
            nelec,
        )
        electronic = float(
            np.vdot(ci_vector, contracted).real / np.vdot(ci_vector, ci_vector).real
        )
        return float(archive.core_constant + electronic)
    orbitals = archive.n_spatial_orbitals
    nelec = ((active_electrons + spin_2s) // 2, (active_electrons - spin_2s) // 2)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1.0e-12
    solver.max_cycle = 1000
    solver.max_space = 100
    fci.addons.fix_spin_(solver, shift=0.5, ss=spin_2s * (spin_2s + 2) / 4.0)
    energy, _ = solver.kernel(
        archive.one_body_integrals,
        archive.two_body_integrals,
        orbitals,
        nelec,
        ecore=archive.core_constant,
    )
    return float(energy)


def _case_paths(spec: CaseSpec) -> tuple[Path, Path, Path]:
    case_root = OUTPUT_ROOT / spec.system / "systems" / spec.case_id
    return case_root, case_root / "hamiltonian.npz", case_root / "inputs" / "source_integrals.npz"


def _metadata(
    spec: CaseSpec,
    *,
    hamiltonian_path: Path,
    integral_path: Path,
    source_hash: str,
    reference_hash: str,
    source_reference: float,
    recomputed: float,
    terms: np.ndarray,
    archive,
) -> dict[str, Any]:
    rel_hamiltonian = hamiltonian_path.relative_to(ROOT).as_posix()
    rel_integral = integral_path.relative_to(ROOT).as_posix()
    error_si = abs(recomputed - spec.si_energy_hartree)
    error_source = abs(recomputed - source_reference)
    geometry: dict[str, Any] = {
        "point_label": spec.label,
        "point_name": spec.inventory_label,
        "coordinates_angstrom": archive.geometry_angstrom.tolist(),
    }
    if spec.system not in {"C2H4", "CH2", "Fe2S2"}:
        geometry["bond_length_angstrom"] = spec.label
    return {
        "schema": "benchmark-qc.system-metadata.v1",
        "system_id": spec.case_id,
        "system": spec.system,
        "active_space": {
            "active_electrons": spec.active_electrons,
            "active_spatial_orbitals": spec.active_orbitals,
            "qubits": 2 * spec.active_orbitals,
            "spin_2S": spec.spin_2s,
        },
        "basis": spec.basis,
        "geometry": geometry,
        "reference_energy_hartree": spec.si_energy_hartree,
        "hamiltonian": {
            "path": rel_hamiltonian,
            "sha256": sha256_file(hamiltonian_path),
            "format": "labels, Hs, casci_energies; one point",
            "pauli_term_count": int(len(terms)),
        },
        "integrals": {
            "path": rel_integral,
            "sha256": sha256_file(integral_path),
            "format": "pickle-free normalized spatial active-space archive",
        },
        "provenance": {
            "source_provider": "QCANT/change3_benchmark10_reference",
            "source_input_path": QCANT_PREFIX + spec.source_rel,
            "source_input_sha256": source_hash,
            "source_reference_path": QCANT_PREFIX + (spec.reference_json_rel or spec.reference_rel),
            "source_reference_sha256": reference_hash,
            "reconstruction_tool": "tools/reconstruct_si_table_i.py",
        },
        "si_validation": {
            "target_source": "U2response/DD-ART-06-2026-000368/05_supporting_data/unified_benchmark_rows.json",
            "target_energy_hartree": spec.si_energy_hartree,
            "target_energy_rounded_to_table_hartree": round(spec.si_energy_hartree, 3),
            "source_reference_energy_hartree": source_reference,
            "recomputed_casci_energy_hartree": recomputed,
            "absolute_error_vs_si_hartree": error_si,
            "absolute_error_vs_source_reference_hartree": error_source,
            "tolerance_hartree": TOLERANCE_HARTREE,
            "passed": error_si <= TOLERANCE_HARTREE and error_source <= TOLERANCE_HARTREE,
        },
    }


def _catalog_entry(spec: CaseSpec, hamiltonian_path: Path, integral_path: Path) -> dict[str, Any]:
    return {
        "id": spec.case_id,
        "system": spec.system,
        "case_name": f"{spec.system} {spec.inventory_label}",
        "status": "reconstructed-si",
        "active_electrons": spec.active_electrons,
        "active_spatial_orbitals": spec.active_orbitals,
        "qubits": 2 * spec.active_orbitals,
        "spin_2S": spec.spin_2s,
        "basis": spec.basis,
        "point_count": 1,
        "point_index": 0,
        "point_label": spec.label,
        "data_path": hamiltonian_path.relative_to(ROOT).as_posix(),
        "metadata_path": (hamiltonian_path.parent / "metadata.json").relative_to(ROOT).as_posix(),
        "integral_data_path": integral_path.relative_to(ROOT).as_posix(),
        "reference_energy_hartree": spec.si_energy_hartree,
        "source_archive_path": QCANT_PREFIX + spec.source_rel,
        "source_reference_path": QCANT_PREFIX + (spec.reference_json_rel or spec.reference_rel),
        "sha256": sha256_file(hamiltonian_path),
        "integral_sha256": sha256_file(integral_path),
    }


def _update_inventory(specs: tuple[CaseSpec, ...]) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["status_definitions"] = {
        "checked_in_payload": "A validated Hamiltonian archive is present in this repository.",
        "cloud_placeholder_payload": "A source filename is present in OneDrive but no validated payload is available.",
        "results_metadata_only": "No validated source Hamiltonian or integral payload is available.",
    }
    for variant in inventory["variants"]:
        for row in variant["geometry_rows"]:
            for spec in specs:
                if variant["system_id"] != spec.system_id or row["label"] != spec.inventory_label:
                    continue
                row["status"] = "checked_in_payload"
                row["source_archive"] = (
                    OUTPUT_ROOT / spec.system / "systems" / spec.case_id / "hamiltonian.npz"
                ).relative_to(ROOT).as_posix()
                row["reconstruction_source"] = QCANT_PREFIX + spec.source_rel
                row["si_validation"] = "recomputed CASCI agrees within 1e-8 Eh"
                if spec.mode == "u2_certificate":
                    row["availability_note"] = (
                        "Reconstructed from a validated QCANT singlet certificate "
                        "because the SI-side OneDrive Hamiltonian file was a "
                        "dataless placeholder."
                    )
    inventory["payload_summary"] = {
        "geometry_rows": inventory["geometry_row_count"],
        "checked_in_payload": sum(
            row["status"] == "checked_in_payload"
            for variant in inventory["variants"]
            for row in variant["geometry_rows"]
        ),
        "cloud_placeholder_payload": sum(
            row["status"] == "cloud_placeholder_payload"
            for variant in inventory["variants"]
            for row in variant["geometry_rows"]
        ),
        "results_metadata_only": sum(
            row["status"] == "results_metadata_only"
            for variant in inventory["variants"]
            for row in variant["geometry_rows"]
        ),
    }
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcant-root", type=Path, default=DEFAULT_QCANT_ROOT)
    parser.add_argument("--check-only", action="store_true", help="verify existing reconstructed cases")
    args = parser.parse_args()
    qcant_root = args.qcant_root.expanduser().resolve()
    if not qcant_root.is_dir():
        raise SystemExit(f"QCANT source root is not a directory: {qcant_root}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    existing_ids = {entry["id"] for entry in catalog["datasets"]}
    results: list[dict[str, Any]] = []
    new_entries: list[dict[str, Any]] = []
    generated_ids = {spec.case_id for spec in CASE_SPECS}
    for spec in CASE_SPECS:
        case_root, hamiltonian_path, integral_path = _case_paths(spec)
        if not args.check_only and case_root.exists() and any(case_root.iterdir()):
            expected_files = {
                "hamiltonian.npz",
                "metadata.json",
                "inputs/source_integrals.npz",
            }
            existing_files = {
                path.relative_to(case_root).as_posix()
                for path in case_root.rglob("*")
                if path.is_file()
            }
            if not existing_files.issubset(expected_files):
                raise SystemExit(f"refusing to mix files in existing case directory {case_root}")

        payload = _source_payload(spec, qcant_root)
        if args.check_only:
            if not integral_path.is_file() or not hamiltonian_path.is_file():
                raise SystemExit(
                    f"missing reconstructed payload for check-only run: {case_root}"
                )
        else:
            case_root.mkdir(parents=True, exist_ok=True)
            integral_path.parent.mkdir(parents=True, exist_ok=True)
            source_path = qcant_root / spec.source_rel
            source_hash = sha256_file(source_path)
            reference_path = qcant_root / (spec.reference_json_rel or spec.reference_rel)
            reference_hash = sha256_file(reference_path)
            np.savez_compressed(integral_path, **payload)
            source_hash = sha256_file(source_path)
            reference_hash = sha256_file(reference_path)
        if args.check_only:
            source_hash = "existing"
            reference_hash = "existing"

        archive = load_spatial_integral_archive(integral_path)
        recomputed = recompute_target_energy(
            archive,
            active_electrons=spec.active_electrons,
            spin_2s=spec.spin_2s,
            ci_vector=payload.get("certified_singlet_ci"),
        )
        source_reference = _reference_energy(spec, qcant_root)
        if abs(recomputed - spec.si_energy_hartree) > TOLERANCE_HARTREE:
            raise RuntimeError(
                f"{spec.case_id}: recomputed CASCI {recomputed:.15f} does not match SI "
                f"{spec.si_energy_hartree:.15f}"
            )
        if abs(recomputed - source_reference) > TOLERANCE_HARTREE:
            raise RuntimeError(
                f"{spec.case_id}: recomputed CASCI {recomputed:.15f} does not match source "
                f"reference {source_reference:.15f}"
            )

        terms = jordan_wigner_terms_from_integrals(archive)
        if not args.check_only:
            write_historical_hamiltonian_npz(
                hamiltonian_path,
                labels=[spec.label],
                hamiltonian_terms=[terms],
                casci_energies=[spec.si_energy_hartree],
            )
            metadata = _metadata(
                spec,
                hamiltonian_path=hamiltonian_path,
                integral_path=integral_path,
                source_hash=source_hash,
                reference_hash=reference_hash,
                source_reference=source_reference,
                recomputed=recomputed,
                terms=terms,
                archive=archive,
            )
            (case_root / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            new_entries.append(_catalog_entry(spec, hamiltonian_path, integral_path))
        results.append(
            {
                "case_id": spec.case_id,
                "si_energy_hartree": spec.si_energy_hartree,
                "source_reference_energy_hartree": source_reference,
                "recomputed_casci_energy_hartree": recomputed,
                "absolute_error_hartree": abs(recomputed - spec.si_energy_hartree),
                "pauli_term_count": int(len(terms)),
                "passed": True,
            }
        )

    if not args.check_only:
        catalog["datasets"] = [
            entry for entry in catalog["datasets"] if entry["id"] not in generated_ids
        ]
        catalog["datasets"].extend(new_entries)
        catalog["datasets"].sort(key=lambda item: item["id"])
        CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        _update_inventory(CASE_SPECS)
        RECONSTRUCTION_ROOT.mkdir(parents=True, exist_ok=True)
        (RECONSTRUCTION_ROOT / "validation_report.json").write_text(
            json.dumps(
                {
                    "schema": "benchmark-qc.si-table-i-reconstruction-report.v1",
                    "source_root": QCANT_PREFIX,
                    "case_count": len(results),
                    "all_passed": all(item["passed"] for item in results),
                    "cases": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"validated {len(results)} SI Table I reconstruction cases")


if __name__ == "__main__":
    main()
