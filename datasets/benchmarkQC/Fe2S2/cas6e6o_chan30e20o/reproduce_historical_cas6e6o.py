#!/usr/bin/env python3
"""Reproduce all historical reduced-space Fe2S2 constructions faithfully.

This is a clean-room, checksum-bound implementation of the Hamiltonian setup
in ``provenance/vi66cycle_historical_source.py``.  It deliberately preserves the
original default-RHF route for CAS(4e,4o), CAS(6e,6o), and CAS(8e,6o): no
Active Space Finder call is made here.  The ASF calculation is the separate
``generate_asf_cas6e6o.py`` workflow.

The candidate is marked accepted only if the physical singlet, residual, RDM,
source-integrity, and published-energy gates all pass.  A rejected candidate
is still written for diagnosis, but cannot be promoted or cataloged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from pyscf import ao2mo, mcscf


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workflow_common import (  # noqa: E402
    DEFAULT_CAS6_TOLERANCE_HARTREE,
    HISTORICAL_SOURCE_SHA256,
    atomic_json,
    candidate_gates,
    correlation_descriptors,
    load_parent_fcidump,
    make_integral_mean_field,
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
HISTORICAL_SPACES = {
    "cas4e4o": {
        "variant": "historical_cas4e4o",
        "dataset_id": "fe2s2_chan30e20o_historical_rhf_cas4e4o",
        "active_electrons": 4,
        "active_orbitals": 4,
        "published_energy_hartree": -116.154749,
    },
    "cas6e6o": {
        "variant": "historical_cas6e6o",
        "dataset_id": "fe2s2_chan30e20o_historical_rhf_cas6e6o",
        "active_electrons": 6,
        "active_orbitals": 6,
        "published_energy_hartree": -116.156259,
    },
    "cas8e6o": {
        "variant": "historical_cas8e6o",
        "dataset_id": "fe2s2_chan30e20o_historical_rhf_cas8e6o",
        "active_electrons": 8,
        "active_orbitals": 6,
        "published_energy_hartree": -116.158572,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fcidump", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=tuple(HISTORICAL_SPACES),
        default=tuple(HISTORICAL_SPACES),
        help="Historical spaces to generate; default: all three paper spaces.",
    )
    parser.add_argument(
        "--execution-site",
        required=True,
        choices=("desktop", "talon"),
        help="Chemistry is permitted only on desktop or in a Talon Slurm job.",
    )
    parser.add_argument(
        "--energy-tolerance-hartree",
        "--cas6-energy-tolerance-hartree",
        dest="energy_tolerance_hartree",
        type=float,
        default=DEFAULT_CAS6_TOLERANCE_HARTREE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    if not (
        0.0
        < args.energy_tolerance_hartree
        <= DEFAULT_CAS6_TOLERANCE_HARTREE
    ):
        raise ValueError(
            "Historical energy tolerance must be positive and no larger than "
            f"{DEFAULT_CAS6_TOLERANCE_HARTREE:.1e} Ha"
        )
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

    # Replay the shared historical frame exactly once. Each paper model then uses
    # PySCF's default CASCI partition for its own (electrons, orbitals) pair.
    mean_field = make_integral_mean_field(
        h1_parent, eri_parent, parent_ecore, run_rhf=True
    )
    summaries = []
    every_accepted = True
    for space_name in args.spaces:
        definition = HISTORICAL_SPACES[space_name]
        active_electrons = int(definition["active_electrons"])
        active_orbitals = int(definition["active_orbitals"])
        published_energy = float(definition["published_energy_hartree"])
        candidate_output = output / space_name
        candidate_output.mkdir()
        casci = mcscf.CASCI(mean_field, active_orbitals, active_electrons)
        casci.fix_spin(ss=0.0)
        h1_active, active_ecore = casci.get_h1eff(casci.mo_coeff)
        eri_active = ao2mo.restore(
            1, casci.get_h2eff(casci.mo_coeff), active_orbitals
        )
        energy, ci, solver_record = solve_active_singlet(
            np.asarray(h1_active),
            np.asarray(eri_active),
            float(active_ecore),
            faithful_direct_spin1=True,
            active_orbitals=active_orbitals,
            active_electrons=active_electrons,
        )
        gamma, gamma2, cumulant, rdm_checks = spin_orbital_rdms(
            ci,
            active_orbitals=active_orbitals,
            active_electrons=active_electrons,
        )
        reconstructed = rdm_energy(
            gamma, gamma2, np.asarray(h1_active), np.asarray(eri_active), float(active_ecore)
        )
        rdm_energy_error = abs(reconstructed - energy)
        descriptors = correlation_descriptors(ci, cumulant)
        classical_controls, control_arrays = run_classical_controls(
            np.asarray(h1_active),
            np.asarray(eri_active),
            float(active_ecore),
            active_electrons=active_electrons,
        )
        active_start = int(casci.ncore)
        active_indices = np.arange(
            active_start, active_start + active_orbitals, dtype=np.int64
        )
        expected_start = (30 - active_electrons) // 2
        reference = np.asarray(
            [True] * active_electrons
            + [False] * (2 * active_orbitals - active_electrons)
        )
        artifacts = write_candidate_artifacts(
            output_directory=candidate_output,
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
                "historical_active_indices_zero_based": active_indices,
                **control_arrays,
            },
        )
        gates = candidate_gates(
            energy=energy,
            published_energy=published_energy,
            energy_tolerance=args.energy_tolerance_hartree,
            solver=solver_record,
            rdm_checks=rdm_checks,
            rdm_energy_error=rdm_energy_error,
            additional={
                "historical_source_checksum": historical_hash == HISTORICAL_SOURCE_SHA256,
                "historical_default_rhf_replay_completed": bool(
                    np.isfinite(float(mean_field.e_tot))
                ),
                "historical_default_partition": bool(
                    np.array_equal(
                        active_indices,
                        np.arange(expected_start, expected_start + active_orbitals),
                    )
                ),
            },
        )
        accepted = all(gates.values())
        every_accepted = every_accepted and accepted
        report = {
            "schema": "benchmark-qc.fe2s2-candidate.v1",
            "dataset_id": definition["dataset_id"],
            "variant": definition["variant"],
            "status": "accepted" if accepted else "rejected",
            "target": {
                "active_electrons": active_electrons,
                "active_spatial_orbitals": active_orbitals,
                "qubits": 2 * active_orbitals,
                "spin_2S": 0,
            },
            "method": {
                "name": "historical nonstationary default-RHF/CASCI partition",
                "orbital_frame_status": "historical_nonstationary_default_rhf",
                "stationary_rhf_claim": False,
                "active_space_finder_used": False,
                "recipe": (
                    "shared checksum-bound default PySCF RHF replay on the Chan "
                    f"FCIDUMP, CASCI({active_electrons},{active_orbitals}) default "
                    "orbital partition, then direct_spin1 FCI"
                ),
                "historical_source": {
                    "path": "provenance/vi66cycle_historical_source.py",
                    "sha256": historical_hash,
                    "scope_note": (
                        "The recovered file explicitly contains CAS(6e,6o); the "
                        "same default-partition construction defines the paper's "
                        "CAS(4e,4o) and CAS(8e,6o) sibling models."
                    ),
                },
                "rhf_converged": bool(mean_field.converged),
                "rhf_energy_hartree": float(mean_field.e_tot),
                "rhf_occupied_virtual_gradient_norm": float(
                    np.linalg.norm(
                        mean_field.get_grad(
                            mean_field.mo_coeff,
                            mean_field.mo_occ,
                        )
                    )
                ),
                "historical_nonstationarity_note": (
                    "The supplied source used the default RHF cycle limit without "
                    "a convergence check. This faithfully preserved frame is not a "
                    "conventional stationary-RHF or ASF-selected frame."
                ),
                "frozen_core_indices_in_rhf_frame_zero_based": list(range(active_start)),
                "active_indices_in_rhf_frame_zero_based": active_indices.tolist(),
                "external_indices_in_rhf_frame_zero_based": list(
                    range(active_start + active_orbitals, 20)
                ),
            },
            "parent": parent_record,
            "acceptance": {
                "published_energy_hartree": published_energy,
                "energy_tolerance_hartree": args.energy_tolerance_hartree,
            },
            "results": {
                "casci_energy_hartree": energy,
                "difference_from_published_hartree": energy - published_energy,
                "published_value_matches_by_truncation_to_6_decimals": (
                    math.trunc(energy * 1.0e6) / 1.0e6 == published_energy
                ),
                "rdm_reconstructed_energy_hartree": reconstructed,
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
        report_path = candidate_output / "candidate_report.json"
        atomic_json(report_path, report)
        summaries.append(
            {
                "space": space_name,
                "status": report["status"],
                "casci_energy_hartree": energy,
                "difference_from_published_hartree": energy - published_energy,
                "report": str(report_path),
            }
        )
    print(
        json.dumps({"candidates": summaries}, indent=2, sort_keys=True)
    )
    return 0 if every_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
