#!/usr/bin/env python3
"""Generate an ASF-selected CAS(6e,6o) from the Chan CAS(30e,20o) FCIDUMP.

The parent DMRG calculation is performed directly in the 20-orbital frame
stored in the checksum-pinned FCIDUMP.  Its one- and two-body RDMs are passed
to the public Active Space Finder pair-information selector.  The resulting
strict fixed-size CAS(6e,6o) is solved independently with spin-adapted FCI.

This route is intentionally distinct from the historical default-RHF recipe
in ``reproduce_historical_cas6e6o.py``.  No orbital set is selected by fitting
the reported CAS(6e,6o) energy; that value is used only as a final acceptance
gate after ASF has made its selection.
"""

from __future__ import annotations

import argparse
import json
import os
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
    DEFAULT_PARENT_DMRG_TOLERANCE_HARTREE,
    PARENT_NELEC,
    PARENT_NORB,
    PUBLISHED_CAS6_ENERGY_HARTREE,
    PUBLISHED_PARENT_DMRG_ENERGY_HARTREE,
    TARGET_NELEC,
    TARGET_NORB,
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
    solve_active_singlet,
    spin_orbital_rdms,
    write_candidate_artifacts,
)


DATASET_ID = "fe2s2_chan30e20o_asf_dmrg_cas6e6o"
DEFAULT_PARENT = HERE / "parent" / "chan_fe2s2_cas30e20o.FCIDUMP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fcidump", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-site", required=True, choices=("desktop", "talon")
    )
    parser.add_argument(
        "--block-executable",
        type=Path,
        required=True,
        help="Plain path to the configured Block2/Block executable used by pyscf-dmrgscf.",
    )
    parser.add_argument(
        "--mpi-prefix",
        default="",
        help="Optional pyscf-dmrgscf MPI prefix, for example 'mpirun -np 8'.",
    )
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--max-bond-dimension", type=int, default=1000)
    parser.add_argument("--dmrg-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-gb", type=float, default=8.0)
    parser.add_argument(
        "--fallback-unfiltered",
        action="store_true",
        help=(
            "If strict ASF fixed-size selection has no CAS(6e,6o), explicitly "
            "retry the documented ASF unfiltered candidate pool. Strict "
            "selection remains the default."
        ),
    )
    parser.add_argument(
        "--cas6-energy-tolerance-hartree",
        type=float,
        default=DEFAULT_CAS6_TOLERANCE_HARTREE,
    )
    parser.add_argument(
        "--parent-dmrg-energy-tolerance-hartree",
        type=float,
        default=DEFAULT_PARENT_DMRG_TOLERANCE_HARTREE,
    )
    return parser.parse_args()


def _configure_dmrg(args: argparse.Namespace):
    try:
        from pyscf import dmrgscf
    except ImportError as error:
        raise RuntimeError(
            "pyscf-dmrgscf is required for the Chan parent calculation"
        ) from error
    executable = args.block_executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Block executable is not executable: {executable}")
    dmrgscf.settings.BLOCKEXE = str(executable)
    dmrgscf.settings.MPIPREFIX = str(args.mpi_prefix)
    return dmrgscf, executable


def _run_parent_dmrg(
    mean_field,
    *,
    dmrgscf,
    scratch: Path,
    max_bond_dimension: int,
    tolerance: float,
    threads: int,
    memory_gb: float,
):
    scratch.mkdir(parents=True, exist_ok=True)
    casci = mcscf.CASCI(mean_field, PARENT_NORB, PARENT_NELEC)
    solver = dmrgscf.DMRGCI(
        mol=mean_field.mol,
        maxM=max_bond_dimension,
        tol=tolerance,
        num_thrds=threads,
        memory=memory_gb,
    )
    solver.nroots = 1
    solver.scratchDirectory = str(scratch)
    solver.runtimeDir = str(scratch)
    casci.fcisolver = solver
    casci.kernel(mo_coeff=np.eye(PARENT_NORB))
    if not np.isfinite(float(casci.e_tot)):
        raise RuntimeError("parent DMRG returned a non-finite energy")
    rdm1, rdm2 = dmrgscf.DMRGCI.make_rdm12(
        solver, 0, PARENT_NORB, PARENT_NELEC
    )
    return casci, np.asarray(rdm1), np.asarray(rdm2)


def _asf_selection(
    mean_field,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    *,
    allow_unfiltered_fallback: bool,
):
    """Use ASF's public pair-information implementation on parent DMRG RDMs."""

    from asf.asfbase import ASFBase
    from asf.filters import ActiveSpaceSelectionError
    from asf.pairinfo import cumulant2b
    from asf.utility import orbdens_from_rdm, rdm1s_from_rdm12

    rdm1a, rdm1b = rdm1s_from_rdm12(PARENT_NELEC, 0.0, rdm1, rdm2)
    orbital_density = orbdens_from_rdm(rdm1a, rdm1b, 0.5 * rdm2)
    cumulant = cumulant2b(rdm1a, rdm1b, rdm2)
    diagonal_cumulant = np.empty((PARENT_NORB, PARENT_NORB), dtype=float)
    for p in range(PARENT_NORB):
        for q in range(PARENT_NORB):
            diagonal_cumulant[p, q] = float(cumulant[p, p, q, q])

    class RDMBackedASF(ASFBase):
        def __init__(self):
            super().__init__(
                mean_field.mol,
                np.eye(PARENT_NORB),
                nel=PARENT_NELEC,
                mo_list=np.arange(PARENT_NORB),
            )

        def calculate(self) -> None:
            return None

        def one_orbital_density(self, root: int = 0) -> np.ndarray:
            if root != 0:
                raise ValueError("only the parent DMRG ground state is available")
            return orbital_density.copy()

        def diagonal_cumulant(
            self, root: int = 0, full_space: bool = True
        ) -> np.ndarray:
            if root != 0:
                raise ValueError("only the parent DMRG ground state is available")
            return diagonal_cumulant.copy()

        def rdm1s(self, root: int = 0) -> np.ndarray:
            if root != 0:
                raise ValueError("only the parent DMRG ground state is available")
            return np.asarray([rdm1a, rdm1b])

        def rdm2(self, root: int = 0) -> np.ndarray:
            if root != 0:
                raise ValueError("only the parent DMRG ground state is available")
            return rdm2.copy()

    selector = RDMBackedASF()
    # First attempt strict selection unconditionally.  The fallback is a
    # scientifically distinct protocol and is reached only after an explicit
    # command-line opt-in; its triggering exception is retained verbatim.
    strict_error = None
    try:
        selected = selector.find_one_sized(
            root=0,
            norb=TARGET_NORB,
            nel=TARGET_NELEC,
            fallback_unfiltered=False,
        )
        selection_mode = "strict_fixed_size"
    except ActiveSpaceSelectionError as error:
        strict_error = f"{type(error).__name__}: {error}"
        if not allow_unfiltered_fallback:
            raise
        selected = selector.find_one_sized(
            root=0,
            norb=TARGET_NORB,
            nel=TARGET_NELEC,
            fallback_unfiltered=True,
        )
        selection_mode = "explicit_unfiltered_fallback"
    if int(selected.norb) != TARGET_NORB or int(selected.nel) != TARGET_NELEC:
        raise RuntimeError(
            f"ASF returned CAS({selected.nel}e,{selected.norb}o), not CAS(6e,6o)"
        )
    return (
        selected,
        orbital_density,
        diagonal_cumulant,
        {
            "selection_mode": selection_mode,
            "strict_default": True,
            "unfiltered_fallback_authorized": bool(allow_unfiltered_fallback),
            "unfiltered_fallback_used": (
                selection_mode == "explicit_unfiltered_fallback"
            ),
            "strict_selection_error": strict_error,
        },
    )


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.max_bond_dimension <= 0 or args.threads <= 0 or args.memory_gb <= 0.0:
        raise ValueError("DMRG resources must be positive")
    if args.dmrg_tolerance <= 0.0:
        raise ValueError("DMRG tolerance must be positive")
    if not (
        0.0
        < args.cas6_energy_tolerance_hartree
        <= DEFAULT_CAS6_TOLERANCE_HARTREE
    ):
        raise ValueError(
            "CAS(6e,6o) energy tolerance must be positive and no larger than "
            f"{DEFAULT_CAS6_TOLERANCE_HARTREE:.1e} Ha"
        )
    if not (
        0.0
        < args.parent_dmrg_energy_tolerance_hartree
        <= DEFAULT_PARENT_DMRG_TOLERANCE_HARTREE
    ):
        raise ValueError(
            "parent DMRG energy tolerance must be positive and no larger than "
            f"{DEFAULT_PARENT_DMRG_TOLERANCE_HARTREE:.1e} Ha"
        )
    site_record = require_remote_execution_site(args.execution_site)
    output = prepare_empty_output_directory(args.output_dir)
    h1_parent, eri_parent, parent_ecore, parent_record = load_parent_fcidump(
        args.parent_fcidump
    )
    mean_field = make_integral_mean_field(
        h1_parent, eri_parent, parent_ecore, run_rhf=False
    )
    dmrgscf, block_executable = _configure_dmrg(args)
    parent_casci, parent_rdm1, parent_rdm2 = _run_parent_dmrg(
        mean_field,
        dmrgscf=dmrgscf,
        scratch=args.scratch_dir.resolve(),
        max_bond_dimension=args.max_bond_dimension,
        tolerance=args.dmrg_tolerance,
        threads=args.threads,
        memory_gb=args.memory_gb,
    )
    (
        selected,
        parent_orbital_density,
        parent_diagonal_cumulant,
        selection_record,
    ) = _asf_selection(
        mean_field,
        parent_rdm1,
        parent_rdm2,
        allow_unfiltered_fallback=args.fallback_unfiltered,
    )
    selected_indices = np.asarray(selected.mo_list, dtype=np.int64)

    reduced = mcscf.CASCI(mean_field, TARGET_NORB, TARGET_NELEC)
    reduced.fix_spin(ss=0.0)
    sorted_mo = reduced.sort_mo(
        caslst=selected_indices.tolist(),
        mo_coeff=np.asarray(selected.mo_coeff),
        base=0,
    )
    h1_active, active_ecore = reduced.get_h1eff(sorted_mo)
    eri_active = ao2mo.restore(
        1, reduced.get_h2eff(sorted_mo), TARGET_NORB
    )
    energy, ci, solver_record = solve_active_singlet(
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
        faithful_direct_spin1=False,
    )
    gamma, gamma2, cumulant, rdm_checks = spin_orbital_rdms(ci)
    reconstructed = rdm_energy(
        gamma,
        gamma2,
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
    )
    rdm_energy_error = abs(reconstructed - energy)
    descriptors = correlation_descriptors(ci, cumulant)
    classical_controls, control_arrays = run_classical_controls(
        np.asarray(h1_active),
        np.asarray(eri_active),
        float(active_ecore),
    )

    permutation = np.argmax(np.abs(sorted_mo), axis=0).astype(np.int64)
    ncore = int(reduced.ncore)
    core_indices = permutation[:ncore]
    sorted_active_indices = permutation[ncore : ncore + TARGET_NORB]
    external_indices = permutation[ncore + TARGET_NORB :]
    reference = np.asarray([True] * TARGET_NELEC + [False] * TARGET_NELEC)
    artifacts = write_candidate_artifacts(
        output_directory=output,
        h1=np.asarray(h1_active),
        eri=np.asarray(eri_active),
        ecore=float(active_ecore),
        active_indices=selected_indices,
        reference_determinant=reference,
        energy=energy,
        ci=ci,
        gamma=gamma,
        gamma2=gamma2,
        cumulant=cumulant,
        extra_arrays={
            "parent_dmrg_one_rdm": parent_rdm1,
            "parent_dmrg_two_rdm": parent_rdm2,
            "parent_asf_one_orbital_density": parent_orbital_density,
            "parent_asf_diagonal_cumulant": parent_diagonal_cumulant,
            "parent_to_sorted_mo_coeff": sorted_mo,
            "asf_selected_parent_indices_zero_based": selected_indices,
            "frozen_core_parent_indices_zero_based": core_indices,
            "external_parent_indices_zero_based": external_indices,
            **control_arrays,
        },
    )
    parent_energy = float(parent_casci.e_tot)
    parent_rdm_trace_error = abs(float(np.trace(parent_rdm1)) - PARENT_NELEC)
    parent_orbdens_normalization_error = float(
        np.max(np.abs(np.sum(parent_orbital_density, axis=1) - 1.0))
    )
    gates = candidate_gates(
        energy=energy,
        energy_tolerance=args.cas6_energy_tolerance_hartree,
        solver=solver_record,
        rdm_checks=rdm_checks,
        rdm_energy_error=rdm_energy_error,
        additional={
            "asf_fixed_size_selection_completed": True,
            "asf_fallback_use_was_explicit": (
                not selection_record["unfiltered_fallback_used"]
                or selection_record["unfiltered_fallback_authorized"]
            ),
            "asf_returned_cas6e6o": (
                int(selected.nel) == TARGET_NELEC
                and int(selected.norb) == TARGET_NORB
            ),
            "asf_indices_match_sorted_active_block": set(selected_indices.tolist())
            == set(sorted_active_indices.tolist()),
            "parent_dmrg_energy": abs(
                parent_energy - PUBLISHED_PARENT_DMRG_ENERGY_HARTREE
            )
            <= args.parent_dmrg_energy_tolerance_hartree,
            "parent_dmrg_rdm_trace": parent_rdm_trace_error <= 1.0e-6,
            "parent_asf_orbital_density_normalization": (
                parent_orbdens_normalization_error <= 1.0e-8
            ),
        },
    )
    accepted = all(gates.values())
    report = {
        "schema": "benchmark-qc.fe2s2-candidate.v1",
        "dataset_id": DATASET_ID,
        "variant": "asf_dmrg_cas6e6o",
        "status": "accepted" if accepted else "rejected",
        "target": {
            "active_electrons": TARGET_NELEC,
            "active_spatial_orbitals": TARGET_NORB,
            "qubits": 2 * TARGET_NORB,
            "spin_2S": 0,
        },
        "method": {
            "name": "Active Space Finder strict fixed-size CAS(6e,6o)",
            "active_space_finder_used": True,
            "source_orbital_frame": "stored Chan CAS(30e,20o) FCIDUMP orbitals",
            "selection_rule": (
                "ASF find_one_sized(root=0, norb=6, nel=6), using pair "
                "information from the parent singlet DMRG RDMs; strict first, "
                "with the unfiltered pool only after explicit opt-in"
            ),
            "selection": selection_record,
            "selected_parent_indices_zero_based": selected_indices.tolist(),
            "frozen_core_parent_indices_zero_based": core_indices.tolist(),
            "external_parent_indices_zero_based": external_indices.tolist(),
            "dmrg": {
                "block_executable": str(block_executable),
                "mpi_prefix": args.mpi_prefix,
                "max_bond_dimension": args.max_bond_dimension,
                "tolerance": args.dmrg_tolerance,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "scratch_directory": str(args.scratch_dir.resolve()),
            },
        },
        "parent": parent_record,
        "acceptance": {
            "published_cas6_energy_hartree": PUBLISHED_CAS6_ENERGY_HARTREE,
            "cas6_energy_tolerance_hartree": args.cas6_energy_tolerance_hartree,
            "published_parent_dmrg_energy_hartree": (
                PUBLISHED_PARENT_DMRG_ENERGY_HARTREE
            ),
            "parent_dmrg_energy_tolerance_hartree": (
                args.parent_dmrg_energy_tolerance_hartree
            ),
        },
        "results": {
            "parent_dmrg_energy_hartree": parent_energy,
            "parent_dmrg_difference_from_published_hartree": (
                parent_energy - PUBLISHED_PARENT_DMRG_ENERGY_HARTREE
            ),
            "parent_dmrg_rdm_trace_error": parent_rdm_trace_error,
            "parent_asf_orbital_density_normalization_error": (
                parent_orbdens_normalization_error
            ),
            "casci_energy_hartree": energy,
            "difference_from_published_hartree": (
                energy - PUBLISHED_CAS6_ENERGY_HARTREE
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
    report_path = output / "candidate_report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "parent_dmrg_energy_hartree": parent_energy,
                "selected_parent_indices_zero_based": selected_indices.tolist(),
                "casci_energy_hartree": energy,
                "difference_from_published_hartree": (
                    energy - PUBLISHED_CAS6_ENERGY_HARTREE
                ),
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
