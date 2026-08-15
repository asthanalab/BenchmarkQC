#!/usr/bin/env python3
"""Generate the nested CAS(8e,8o) Fe2S2 extension from the pinned parent.

This calculation replays the same checksum-bound default-RHF orbital frame as
the recovered vi66 CAS(6e,6o) workflow.  It then applies PySCF's default
CASCI(8e,8o) partition: parent-frame MOs 11--18 (zero-based), with MOs 0--10
frozen doubly occupied and MO 19 external.  The retained orbital span strictly
contains the historical CAS(6e,6o) span, MOs 12--17.

CAS(8e,8o) was not one of the three active spaces reported in the paper.  The
output is therefore labeled a reconstructed nested extension and is never
validated by fitting a published energy.  Acceptance instead requires the
pinned parent and historical source, the exact nested orbital identity, a physical
singlet FCI residual, consistent RDMs, and a checksum-complete artifact set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyscf
from pyscf import ao2mo, mcscf


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workflow_common import (  # noqa: E402
    HISTORICAL_SOURCE_SHA256,
    atomic_json,
    correlation_descriptors,
    load_parent_fcidump,
    make_integral_mean_field,
    physical_candidate_gates,
    prepare_empty_output_directory,
    rdm_energy,
    require_remote_execution_site,
    run_classical_controls,
    runtime_record,
    sha256_file,
    solve_active_singlet,
    spin_orbital_rdms,
    write_candidate_artifacts,
)


DEFAULT_PARENT = HERE / "parent" / "chan_fe2s2_cas30e20o.FCIDUMP"
HISTORICAL_SOURCE = HERE / "provenance" / "vi66cycle_historical_source.py"
DATASET_ID = "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o"
VARIANT = "nested_cas8e8o"
ACTIVE_ELECTRONS = 8
ACTIVE_ORBITALS = 8
EXPECTED_ACTIVE_INDICES = np.arange(11, 19, dtype=np.int64)
HISTORICAL_CAS6_INDICES = np.arange(12, 18, dtype=np.int64)
EXPECTED_RUNTIME_FRAMES = {
    "primary_pyscf2p4": {
        "pyscf_version": "2.4.0",
        "rhf_energy_hartree": -116.13503262533489,
        "rhf_gradient_norm": 4.845940182351517e-4,
        "mo_coefficients_sha256": (
            "0dfef43644b25561fc3dab2593859eddba350837326117601ca4ef763bab77d8"
        ),
        "require_exact_mo_coefficients_sha256": True,
        "cas6_energy_hartree": -116.15625982795386,
    },
    "crosscheck_pyscf2p11": {
        "pyscf_version": "2.11.0",
        "rhf_energy_hartree": -116.13503262533783,
        "rhf_gradient_norm": 4.8459391288618663e-4,
        "mo_coefficients_sha256": (
            "5cd61c5d37511880d46adb566f16325203a7d48a114d519c47bd666f68062342"
        ),
        "require_exact_mo_coefficients_sha256": False,
        "cas6_energy_hartree": -116.15625982792764,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fcidump", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-role",
        required=True,
        choices=tuple(EXPECTED_RUNTIME_FRAMES),
        help=(
            "Registered same-frame runtime: primary PySCF 2.4 or independent "
            "PySCF 2.11 cross-check."
        ),
    )
    parser.add_argument(
        "--execution-site",
        required=True,
        choices=("desktop", "talon"),
        help="Chemistry is permitted only on desktop or in a Talon Slurm job.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    site_record = require_remote_execution_site(args.execution_site)
    output = prepare_empty_output_directory(args.output_dir)
    h1_parent, eri_parent, parent_ecore, parent_record = load_parent_fcidump(
        args.parent_fcidump
    )

    historical_hash = sha256_file(HISTORICAL_SOURCE)
    if historical_hash != HISTORICAL_SOURCE_SHA256:
        raise RuntimeError(
            "historical source checksum mismatch: "
            f"expected {HISTORICAL_SOURCE_SHA256}, got {historical_hash}"
        )

    mean_field = make_integral_mean_field(
        h1_parent, eri_parent, parent_ecore, run_rhf=True
    )
    expected_frame = EXPECTED_RUNTIME_FRAMES[args.runtime_role]
    parent_gradient = float(
        np.linalg.norm(mean_field.get_grad(mean_field.mo_coeff, mean_field.mo_occ))
    )
    parent_mo_hash = hashlib.sha256(
        np.ascontiguousarray(mean_field.mo_coeff).tobytes()
    ).hexdigest()

    # Reproduce the accepted historical CAS(6e,6o) control in this exact
    # process before deriving the larger nested space.  This binds the new
    # row to the same parent frame rather than merely to a finite RHF energy.
    cas6 = mcscf.CASCI(mean_field, 6, 6)
    cas6.fix_spin(ss=0.0)
    cas6_h1, cas6_ecore = cas6.get_h1eff(cas6.mo_coeff)
    cas6_eri = ao2mo.restore(1, cas6.get_h2eff(cas6.mo_coeff), 6)
    cas6_energy, _cas6_ci, cas6_solver_record = solve_active_singlet(
        np.asarray(cas6_h1),
        np.asarray(cas6_eri),
        float(cas6_ecore),
        faithful_direct_spin1=True,
        active_orbitals=6,
        active_electrons=6,
    )
    casci = mcscf.CASCI(mean_field, ACTIVE_ORBITALS, ACTIVE_ELECTRONS)
    casci.fix_spin(ss=0.0)
    h1_active, active_ecore = casci.get_h1eff(casci.mo_coeff)
    eri_active = ao2mo.restore(
        1, casci.get_h2eff(casci.mo_coeff), ACTIVE_ORBITALS
    )

    energy, ci, solver_record = solve_active_singlet(
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
        # The larger 4,900-determinant sector has a near-degenerate M_S=0
        # Davidson subspace in direct_spin1.  Solve directly in the singlet-
        # adapted space, then certify the returned vector independently with
        # the physical direct_spin1 Hamiltonian and S^2 residuals below.
        faithful_direct_spin1=False,
        active_orbitals=ACTIVE_ORBITALS,
        active_electrons=ACTIVE_ELECTRONS,
    )
    gamma, gamma2, cumulant, rdm_checks = spin_orbital_rdms(
        ci,
        active_orbitals=ACTIVE_ORBITALS,
        active_electrons=ACTIVE_ELECTRONS,
    )
    reconstructed_energy = rdm_energy(
        gamma,
        gamma2,
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
    )
    rdm_energy_error = abs(reconstructed_energy - energy)
    descriptors = correlation_descriptors(ci, cumulant)
    classical_controls, control_arrays = run_classical_controls(
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
        active_electrons=ACTIVE_ELECTRONS,
    )

    active_start = int(casci.ncore)
    active_indices = np.arange(
        active_start, active_start + ACTIVE_ORBITALS, dtype=np.int64
    )
    reference = np.asarray(
        [True] * ACTIVE_ELECTRONS
        + [False] * (2 * ACTIVE_ORBITALS - ACTIVE_ELECTRONS)
    )
    artifacts = write_candidate_artifacts(
        output_directory=output,
        h1=np.asarray(h1_active),
        eri=np.asarray(eri_active),
        ecore=float(active_ecore),
        active_indices=active_indices,
        reference_determinant=reference,
        energy=energy,
        ci=ci,
        gamma=gamma,
        gamma2=gamma2,
        cumulant=cumulant,
        extra_arrays={
            "parent_to_rhf_mo_coeff": np.asarray(mean_field.mo_coeff),
            "rhf_mo_energies": np.asarray(mean_field.mo_energy),
            "rhf_mo_occupations": np.asarray(mean_field.mo_occ),
            "nested_active_indices_zero_based": active_indices,
            **control_arrays,
        },
    )

    nested_contains_cas6 = set(HISTORICAL_CAS6_INDICES).issubset(
        set(active_indices)
    )
    gates = physical_candidate_gates(
        energy=energy,
        solver=solver_record,
        rdm_checks=rdm_checks,
        rdm_energy_error=rdm_energy_error,
        additional={
            "historical_source_checksum": historical_hash == HISTORICAL_SOURCE_SHA256,
            "registered_runtime_version": bool(
                pyscf.__version__ == expected_frame["pyscf_version"]
            ),
            "historical_default_rhf_remains_nonconverged": bool(
                not mean_field.converged
            ),
            "historical_rhf_energy_fingerprint": bool(
                abs(
                    float(mean_field.e_tot)
                    - float(expected_frame["rhf_energy_hartree"])
                )
                <= 1.0e-10
            ),
            "historical_rhf_gradient_fingerprint": bool(
                abs(parent_gradient - float(expected_frame["rhf_gradient_norm"]))
                <= 1.0e-10
            ),
            "historical_rhf_mo_fingerprint_policy": bool(
                not expected_frame["require_exact_mo_coefficients_sha256"]
                or parent_mo_hash == expected_frame["mo_coefficients_sha256"]
            ),
            "historical_cas6_control_energy": bool(
                abs(
                    cas6_energy - float(expected_frame["cas6_energy_hartree"])
                )
                <= 1.0e-10
            ),
            "historical_cas6_control_state": bool(
                cas6_solver_record["converged"]
                and cas6_solver_record[
                    "physical_hamiltonian_residual_norm_hartree"
                ]
                <= 1.0e-9
                and cas6_solver_record["spin_eigenvector_residual_norm"]
                <= 1.0e-8
            ),
            "variational_nesting_energy": bool(
                energy <= cas6_energy + 1.0e-10
            ),
            "nested_default_partition": bool(
                np.array_equal(active_indices, EXPECTED_ACTIVE_INDICES)
            ),
            "strictly_contains_historical_cas6_span": bool(nested_contains_cas6),
            "adds_one_lower_and_one_upper_parent_orbital": bool(
                np.array_equal(
                    np.setdiff1d(active_indices, HISTORICAL_CAS6_INDICES),
                    np.asarray([11, 18], dtype=np.int64),
                )
            ),
        },
    )
    accepted = all(gates.values())
    report = {
        "schema": "benchmark-qc.fe2s2-candidate.v1",
        "dataset_id": DATASET_ID,
        "variant": VARIANT,
        "status": "accepted" if accepted else "rejected",
        "target": {
            "active_electrons": ACTIVE_ELECTRONS,
            "active_spatial_orbitals": ACTIVE_ORBITALS,
            "qubits": 2 * ACTIVE_ORBITALS,
            "spin_2S": 0,
        },
        "method": {
            "name": "nested extension of historical default-RHF/CASCI partition",
            "orbital_frame_status": "historical_nonstationary_default_rhf",
            "stationary_rhf_claim": False,
            "active_space_finder_used": False,
            "historical_paper_space": False,
            "runtime_role": args.runtime_role,
            "recipe": (
                "checksum-bound default PySCF RHF replay on the parent "
                "FCIDUMP, CASCI(8e,8o) default orbital partition, then "
                "singlet-adapted direct_spin0 FCI, followed by independent "
                "direct_spin1 physical-Hamiltonian and S^2 residual checks"
            ),
            "historical_source": {
                "path": "provenance/vi66cycle_historical_source.py",
                "sha256": historical_hash,
                "scope_note": (
                    "The source explicitly defines CAS(6e,6o). CAS(8e,8o) is "
                    "a new nested extension constructed by the same default-"
                    "partition rule; it is not a historical paper row."
                ),
            },
            "rhf_converged": bool(mean_field.converged),
            "rhf_energy_hartree": float(mean_field.e_tot),
            "rhf_occupied_virtual_gradient_norm": parent_gradient,
            "rhf_mo_coefficients_sha256": parent_mo_hash,
            "rhf_mo_coefficients_match_registered_reference": bool(
                parent_mo_hash == expected_frame["mo_coefficients_sha256"]
            ),
            "expected_parent_frame": expected_frame,
            "historical_nonstationarity_note": (
                "The source used the default RHF cycle limit without a "
                "convergence check. The same faithfully replayed, "
                "nonstationary frame is retained for comparability."
            ),
            "frozen_core_indices_in_rhf_frame_zero_based": list(range(active_start)),
            "active_indices_in_rhf_frame_zero_based": active_indices.tolist(),
            "external_indices_in_rhf_frame_zero_based": list(
                range(active_start + ACTIVE_ORBITALS, 20)
            ),
            "nested_parent_space": {
                "historical_cas6_indices_zero_based": HISTORICAL_CAS6_INDICES.tolist(),
                "added_indices_zero_based": [11, 18],
            },
        },
        "parent": parent_record,
        "acceptance": {
            "mode": "reconstructed_nested_extension",
            "runtime_role": args.runtime_role,
            "published_energy_target_used": False,
            "claim_boundary": (
                "Validated numerical extension for a nested active-space "
                "comparison; not one of the historical paper calculations"
            ),
        },
        "results": {
            "historical_cas6_same_process_control": {
                "casci_energy_hartree": cas6_energy,
                "expected_energy_hartree": expected_frame[
                    "cas6_energy_hartree"
                ],
                "difference_hartree": (
                    cas6_energy - float(expected_frame["cas6_energy_hartree"])
                ),
                "solver": cas6_solver_record,
            },
            "casci_energy_hartree": energy,
            "cas8_minus_historical_cas6_energy_hartree": (
                energy - cas6_energy
            ),
            "rdm_reconstructed_energy_hartree": reconstructed_energy,
            "rdm_energy_error_hartree": rdm_energy_error,
            "solver": solver_record,
            "rdm_checks": rdm_checks,
            "correlation_descriptors": descriptors,
            "classical_controls": classical_controls,
        },
        "gates": gates,
        "artifacts": artifacts,
        "runtime": runtime_record(started, site_record),
    }
    report_path = output / "candidate_report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset_id": DATASET_ID,
                "casci_energy_hartree": energy,
                "renyi_h0p25_nats": descriptors["renyi_h0p25_nats"],
                "cumulant_C2": descriptors["cumulant_C2"],
                "n92": descriptors["n92"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
