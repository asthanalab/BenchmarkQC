"""Shared implementation for the checksum-bound Fe2S2 candidate workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pyscf import ao2mo, cc, ci, fci, gto, scf
from pyscf.fci import spin_op
from pyscf.tools import fcidump
from scipy.sparse.linalg import LinearOperator, eigsh


PARENT_NORB = 20
PARENT_NELEC = 30
PARENT_MS2 = 0
TARGET_NORB = 6
TARGET_NELEC = 6
TARGET_NELEC_TUPLE = (3, 3)
PARENT_SHA256 = (
    "95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9"
)
LEGACY_SOURCE_SHA256 = (
    "bddcf0a2f7cb1b6fd354e8418c6a580f1e59a85cb905e7b264fd44f91b621a26"
)
PUBLISHED_CAS6_ENERGY_HARTREE = -116.156259
PUBLISHED_PARENT_DMRG_ENERGY_HARTREE = -116.6056091
DEFAULT_CAS6_TOLERANCE_HARTREE = 1.0e-6
DEFAULT_PARENT_DMRG_TOLERANCE_HARTREE = 5.0e-4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_empty_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new candidate with existing files in {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_remote_execution_site(site: str) -> dict[str, Any]:
    """Enforce the user's desktop-or-Talon calculation rule."""

    hostname = socket.gethostname()
    if site == "desktop":
        if not hostname.lower().startswith("undmedaasthanad"):
            raise RuntimeError(
                f"desktop execution requires host UNDMEDAASTHANAD, got {hostname!r}"
            )
    elif site == "talon":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError(
                "Talon chemistry must run in a Slurm allocation; SLURM_JOB_ID is absent"
            )
    else:
        raise ValueError("execution site must be 'desktop' or 'talon'")
    return {
        "execution_site": site,
        "hostname": hostname,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def load_parent_fcidump(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    source = Path(path).resolve()
    observed_hash = sha256_file(source)
    if observed_hash != PARENT_SHA256:
        raise RuntimeError(
            f"Chan FCIDUMP checksum mismatch: expected {PARENT_SHA256}, got {observed_hash}"
        )
    payload = fcidump.read(str(source), verbose=False)
    header = (
        int(payload["NORB"]),
        int(payload["NELEC"]),
        int(payload["MS2"]),
    )
    if header != (PARENT_NORB, PARENT_NELEC, PARENT_MS2):
        raise RuntimeError(
            f"unexpected Chan FCIDUMP header {header}; "
            f"expected {(PARENT_NORB, PARENT_NELEC, PARENT_MS2)}"
        )
    h1 = np.asarray(payload["H1"], dtype=float)
    eri = np.asarray(ao2mo.restore(1, payload["H2"], PARENT_NORB), dtype=float)
    ecore = float(payload.get("ECORE", 0.0))
    return h1, eri, ecore, {
        "path": str(source),
        "sha256": observed_hash,
        "norb": PARENT_NORB,
        "nelec": PARENT_NELEC,
        "ms2": PARENT_MS2,
        "ecore_hartree": ecore,
    }


def make_integral_mean_field(
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    *,
    run_rhf: bool,
) -> Any:
    """Create the PySCF integral-container used by both documented routes."""

    molecule = gto.Mole()
    molecule.nelectron = PARENT_NELEC
    molecule.spin = PARENT_MS2
    molecule.incore_anyway = True
    molecule.verbose = 0
    molecule.build()
    mean_field = scf.RHF(molecule)
    mean_field.get_hcore = lambda *unused: h1
    mean_field.get_ovlp = lambda *unused: np.eye(PARENT_NORB)
    mean_field._eri = ao2mo.restore(8, eri, PARENT_NORB)
    # This instance-level override reproduces the historical source exactly.
    mean_field.energy_nuc = lambda *unused: float(ecore)
    if run_rhf:
        mean_field.kernel()
    else:
        mean_field.mo_coeff = np.eye(PARENT_NORB)
        mean_field.mo_occ = np.asarray([2.0] * 15 + [0.0] * 5)
        mean_field.mo_energy = np.diag(h1).copy()
        density = mean_field.make_rdm1(mean_field.mo_coeff, mean_field.mo_occ)
        mean_field.e_tot = float(mean_field.energy_tot(dm=density))
        mean_field.converged = True
    return mean_field


def solve_active_singlet(
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    *,
    faithful_direct_spin1: bool,
    active_orbitals: int = TARGET_NORB,
    active_electrons: int = TARGET_NELEC,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if active_electrons % 2 or active_electrons > 2 * active_orbitals:
        raise ValueError("target must be a valid closed-shell singlet sector")
    electron_tuple = (active_electrons // 2, active_electrons // 2)
    solver = fci.direct_spin1.FCI() if faithful_direct_spin1 else fci.direct_spin0.FCI()
    # PySCF's default energy-only Davidson criterion can label a larger CI
    # vector converged while its physical eigenvector residual remains too
    # large for a publishable state certificate.  Apply the same explicit
    # residual gate to every active-space size and enforce the requested
    # singlet within the direct_spin1 M_S=0 sector.
    solver.conv_tol = 1.0e-13
    solver.conv_tol_residual = 1.0e-11
    solver.max_cycle = 1000
    solver.max_space = 50
    if faithful_direct_spin1:
        fci.addons.fix_spin_(solver, shift=0.5, ss=0.0)
    electronic_energy, ci = solver.kernel(
        h1,
        eri,
        active_orbitals,
        electron_tuple,
        ecore=0.0,
        nroots=1,
    )
    vector = np.asarray(ci, dtype=float)
    norm = float(np.linalg.norm(vector))
    effective = fci.direct_spin1.absorb_h1e(
        h1, eri, active_orbitals, electron_tuple, 0.5
    )
    contracted = np.asarray(
        fci.direct_spin1.contract_2e(
            effective, vector, active_orbitals, electron_tuple
        )
    )
    initial_residual = float(
        np.linalg.norm(contracted - float(electronic_energy) * vector) / norm
    )
    initial_spin_vector = np.asarray(
        spin_op.contract_ss(vector, active_orbitals, electron_tuple)
    )
    initial_spin_residual = float(np.linalg.norm(initial_spin_vector) / norm)

    refinement_used = bool(
        not solver.converged
        or initial_residual > 1.0e-9
        or initial_spin_residual > 1.0e-8
    )
    refinement_converged = False
    penalized_eigenvalue = None
    if refinement_used:
        shape = vector.shape
        dimension = int(vector.size)
        spin_penalty_shift = 0.5

        def penalized_matvec(flat_vector: np.ndarray) -> np.ndarray:
            trial = np.asarray(flat_vector).reshape(shape)
            h_trial = fci.direct_spin1.contract_2e(
                effective, trial, active_orbitals, electron_tuple
            )
            s2_trial = spin_op.contract_ss(
                trial, active_orbitals, electron_tuple
            )
            return np.asarray(
                h_trial + spin_penalty_shift * s2_trial, dtype=float
            ).ravel()

        operator = LinearOperator(
            (dimension, dimension), matvec=penalized_matvec, dtype=float
        )
        eigenvalues, eigenvectors = eigsh(
            operator,
            k=1,
            which="SA",
            v0=(vector / norm).ravel(),
            tol=1.0e-13,
            maxiter=20000,
            ncv=min(120, dimension - 1),
        )
        penalized_eigenvalue = float(eigenvalues[0])
        vector = np.asarray(eigenvectors[:, 0], dtype=float).reshape(shape)
        vector /= float(np.linalg.norm(vector))
        contracted = np.asarray(
            fci.direct_spin1.contract_2e(
                effective, vector, active_orbitals, electron_tuple
            )
        )
        electronic_energy = float(np.vdot(vector, contracted).real)
        norm = float(np.linalg.norm(vector))
        refinement_converged = True

    residual = float(
        np.linalg.norm(contracted - float(electronic_energy) * vector) / norm
    )
    spin_square, multiplicity = spin_op.spin_square0(
        vector, active_orbitals, electron_tuple
    )
    spin_vector = spin_op.contract_ss(vector, active_orbitals, electron_tuple)
    spin_residual = float(np.linalg.norm(spin_vector) / norm)
    penalized_residual = None
    penalized_energy_consistency = None
    if refinement_used:
        penalized_contracted = contracted + 0.5 * spin_vector
        penalized_residual = float(
            np.linalg.norm(
                penalized_contracted - float(penalized_eigenvalue) * vector
            )
            / norm
        )
        penalized_energy_consistency = abs(
            float(penalized_eigenvalue)
            - (float(electronic_energy) + 0.5 * float(spin_square))
        )
    certified_converged = bool(
        (solver.converged or refinement_converged)
        and residual <= 1.0e-9
        and spin_residual <= 1.0e-8
    )
    return float(electronic_energy + ecore), vector, {
        "solver": type(solver).__name__,
        "energy_convergence_tolerance_hartree": float(solver.conv_tol),
        "residual_convergence_tolerance_hartree": float(
            solver.conv_tol_residual
        ),
        "maximum_davidson_cycles": int(solver.max_cycle),
        "maximum_davidson_subspace": int(solver.max_space),
        "pyscf_davidson_spin_penalty_shift_hartree": (
            0.5 if faithful_direct_spin1 else None
        ),
        "pyscf_solver_converged": bool(solver.converged),
        "initial_physical_hamiltonian_residual_norm_hartree": initial_residual,
        "initial_spin_eigenvector_residual_norm": initial_spin_residual,
        "arpack_refinement_used": refinement_used,
        "arpack_refinement_converged": refinement_converged,
        "arpack_tolerance": 1.0e-13 if refinement_used else None,
        "arpack_maximum_iterations": 20000 if refinement_used else None,
        "arpack_krylov_subspace": (
            min(120, int(vector.size) - 1) if refinement_used else None
        ),
        "penalized_eigenvalue_hartree": penalized_eigenvalue,
        "arpack_spin_penalty_shift_hartree": 0.5 if refinement_used else None,
        "arpack_penalized_residual_norm_hartree": penalized_residual,
        "arpack_penalized_energy_consistency_error_hartree": (
            penalized_energy_consistency
        ),
        "converged": certified_converged,
        "ci_norm": norm,
        "physical_hamiltonian_residual_norm_hartree": residual,
        "spin_square": float(spin_square),
        "multiplicity": float(multiplicity),
        "spin_eigenvector_residual_norm": spin_residual,
    }


def spin_orbital_rdms(
    ci: np.ndarray,
    *,
    active_orbitals: int = TARGET_NORB,
    active_electrons: int = TARGET_NELEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    electron_tuple = (active_electrons // 2, active_electrons // 2)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_spin1.make_rdm12s(
        ci, active_orbitals, electron_tuple
    )
    nso = 2 * active_orbitals
    gamma = np.zeros((nso, nso), dtype=np.result_type(ci, float))
    gamma[:active_orbitals, :active_orbitals] = dm1a.T
    gamma[active_orbitals:, active_orbitals:] = dm1b.T
    gamma2 = np.zeros((nso, nso, nso, nso), dtype=gamma.dtype)
    gamma2[:active_orbitals, :active_orbitals, :active_orbitals, :active_orbitals] = (
        dm2aa.transpose(0, 2, 1, 3)
    )
    gamma2[active_orbitals:, active_orbitals:, active_orbitals:, active_orbitals:] = (
        dm2bb.transpose(0, 2, 1, 3)
    )
    ab = dm2ab.transpose(0, 2, 1, 3)
    gamma2[:active_orbitals, active_orbitals:, :active_orbitals, active_orbitals:] = ab
    gamma2[active_orbitals:, :active_orbitals, :active_orbitals, active_orbitals:] = -ab.transpose(
        1, 0, 2, 3
    )
    gamma2[:active_orbitals, active_orbitals:, active_orbitals:, :active_orbitals] = -ab.transpose(
        0, 1, 3, 2
    )
    gamma2[active_orbitals:, :active_orbitals, active_orbitals:, :active_orbitals] = ab.transpose(
        1, 0, 3, 2
    )
    disconnected = np.einsum("pr,qs->pqrs", gamma, gamma) - np.einsum(
        "ps,qr->pqrs", gamma, gamma
    )
    cumulant = gamma2 - disconnected
    interleaved = np.asarray(
        [
            index
            for spatial in range(active_orbitals)
            for index in (spatial, active_orbitals + spatial)
        ],
        dtype=int,
    )
    gamma = gamma[np.ix_(interleaved, interleaved)]
    gamma2 = gamma2[np.ix_(interleaved, interleaved, interleaved, interleaved)]
    cumulant = cumulant[
        np.ix_(interleaved, interleaved, interleaved, interleaved)
    ]
    contraction = np.einsum("pqrq->pr", gamma2)
    checks = {
        "one_rdm_trace_error": abs(float(np.trace(gamma).real) - active_electrons),
        "two_rdm_contraction_error": float(
            np.linalg.norm(contraction - (active_electrons - 1) * gamma)
        ),
        "two_rdm_hermiticity_error": float(
            np.linalg.norm(gamma2 - gamma2.transpose(2, 3, 0, 1).conj())
        ),
    }
    return gamma, gamma2, cumulant, checks


def rdm_energy(
    gamma: np.ndarray,
    gamma2: np.ndarray,
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
) -> float:
    active_orbitals = int(h1.shape[0])
    alpha = 2 * np.arange(active_orbitals, dtype=int)
    beta = alpha + 1
    one_energy = 0.0
    two_energy = 0.0
    for same_spin in (alpha, beta):
        one_energy += float(
            np.einsum("pq,pq->", h1, gamma[np.ix_(same_spin, same_spin)]).real
        )
        for other_spin in (alpha, beta):
            block = gamma2[np.ix_(same_spin, other_spin, same_spin, other_spin)]
            two_energy += 0.5 * float(
                np.einsum("pqrs,prqs->", eri, block).real
            )
    return float(ecore + one_energy + two_energy)


def correlation_descriptors(ci: np.ndarray, cumulant: np.ndarray) -> dict[str, Any]:
    probabilities = np.abs(np.asarray(ci).ravel()) ** 2
    probabilities /= float(np.sum(probabilities))
    retained = probabilities[probabilities > 1.0e-15]
    retained_mass = float(np.sum(retained))
    retained /= retained_mass
    h025 = math.log(float(np.sum(retained**0.25))) / 0.75
    descending = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(descending)
    n92 = int(np.searchsorted(cumulative, 0.92, side="left") + 1)
    cumulant_squared = float(np.vdot(cumulant, cumulant).real)
    return {
        "renyi_h0p25_nats": float(h025),
        "renyi_probability_cutoff": 1.0e-15,
        "renyi_retained_probability_mass": retained_mass,
        "renyi_retained_support": int(retained.size),
        "n92": n92,
        "largest_determinant_probability": float(descending[0]),
        "cumulant_frobenius_norm": math.sqrt(cumulant_squared),
        "cumulant_C2": 0.25 * cumulant_squared,
        "cumulant_definition": (
            "C2 = (1/4) sum_pqrs |lambda_pqrs|^2 in the full interleaved "
            "spin-orbital basis"
        ),
    }


def run_classical_controls(
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    *,
    active_electrons: int = TARGET_NELEC,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run CISD and CCSD in the same six-orbital Hamiltonian and reference frame.

    These are reported controls, not acceptance gates for the exact CASCI
    Hamiltonian.  In particular, a method-specific fixed-point tolerance must
    not invalidate an otherwise exact, residual-certified Hamiltonian.
    """

    molecule = gto.Mole()
    active_orbitals = int(h1.shape[0])
    molecule.nelectron = active_electrons
    molecule.spin = 0
    molecule.nao = active_orbitals
    molecule.incore_anyway = True
    molecule.verbose = 0
    molecule.build()
    mean_field = scf.RHF(molecule)
    mean_field.get_hcore = lambda *unused: h1
    mean_field.get_ovlp = lambda *unused: np.eye(active_orbitals)
    mean_field._eri = ao2mo.restore(8, eri, active_orbitals)
    mean_field.energy_nuc = lambda *unused: float(ecore)
    occupied_count = active_electrons // 2
    occupations = np.asarray(
        [2.0] * occupied_count + [0.0] * (active_orbitals - occupied_count)
    )
    coefficients = np.eye(active_orbitals)
    density = mean_field.make_rdm1(coefficients, occupations)
    fock = mean_field.get_fock(dm=density)
    occupied_energies, occupied_rotation = np.linalg.eigh(
        fock[:occupied_count, :occupied_count]
    )
    virtual_energies, virtual_rotation = np.linalg.eigh(
        fock[occupied_count:, occupied_count:]
    )
    coefficients[:occupied_count, :occupied_count] = occupied_rotation
    coefficients[occupied_count:, occupied_count:] = virtual_rotation
    density = mean_field.make_rdm1(coefficients, occupations)
    mean_field.mo_coeff = coefficients
    mean_field.mo_occ = occupations
    mean_field.mo_energy = np.concatenate((occupied_energies, virtual_energies))
    mean_field.e_tot = float(mean_field.energy_tot(dm=density))
    mean_field.converged = True
    gradient_norm = float(np.linalg.norm(mean_field.get_grad(coefficients, occupations)))

    cisd = ci.CISD(mean_field)
    cisd.conv_tol = 1.0e-11
    cisd.max_cycle = 500
    cisd.kernel()
    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.conv_tol = 1.0e-12
    coupled_cluster.conv_tol_normt = 1.0e-10
    coupled_cluster.max_cycle = 1000
    coupled_cluster.diis_space = 12
    coupled_cluster.kernel()
    fixed_point_residual = None
    if coupled_cluster.converged:
        updated_t1, updated_t2 = coupled_cluster.update_amps(
            coupled_cluster.t1,
            coupled_cluster.t2,
            coupled_cluster.ao2mo(),
        )
        fixed_point_residual = float(
            math.sqrt(
                np.linalg.norm(updated_t1 - coupled_cluster.t1) ** 2
                + np.linalg.norm(updated_t2 - coupled_cluster.t2) ** 2
            )
        )
    record = {
        "reference_determinant_energy_hartree": float(mean_field.e_tot),
        "reference_occupied_virtual_gradient_norm": gradient_norm,
        "reference_frame": (
            "active Hamiltonian orbital frame with occupied and virtual blocks "
            "semicanonicalized"
        ),
        "cisd_converged": bool(cisd.converged),
        "cisd_energy_hartree": float(cisd.e_tot) if cisd.converged else None,
        "ccsd_converged": bool(coupled_cluster.converged),
        "ccsd_energy_hartree": (
            float(coupled_cluster.e_tot) if coupled_cluster.converged else None
        ),
        "ccsd_fixed_point_update_norm": fixed_point_residual,
        "control_gate_policy": (
            "CISD and CCSD are reported controls and do not gate acceptance of "
            "the exact CASCI Hamiltonian"
        ),
    }
    arrays = {
        "control_reference_mo_coeff": np.asarray(coefficients),
        "control_reference_mo_occupations": np.asarray(occupations),
        "cisd_coefficients": (
            np.asarray(cisd.ci) if cisd.converged else np.asarray([])
        ),
        "ccsd_t1": np.asarray(coupled_cluster.t1),
        "ccsd_t2": np.asarray(coupled_cluster.t2),
    }
    return record, arrays


def physical_candidate_gates(
    *,
    energy: float,
    solver: Mapping[str, Any],
    rdm_checks: Mapping[str, float],
    rdm_energy_error: float,
    additional: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """Return publication-grade numerical gates independent of a paper target."""

    gates = {
        "parent_checksum": True,
        "target_active_space_sector": True,
        "active_space_energy_finite": math.isfinite(float(energy)),
        "active_space_fci_converged": bool(solver["converged"]),
        "active_space_ci_normalized": abs(float(solver["ci_norm"]) - 1.0) <= 1.0e-10,
        "active_space_physical_residual": float(
            solver["physical_hamiltonian_residual_norm_hartree"]
        )
        <= 1.0e-9,
        "active_space_singlet": abs(float(solver["spin_square"])) <= 1.0e-8,
        "active_space_spin_eigenvector": float(solver["spin_eigenvector_residual_norm"])
        <= 1.0e-8,
        "active_space_rdm_trace": float(rdm_checks["one_rdm_trace_error"]) <= 1.0e-9,
        "active_space_rdm_contraction": float(rdm_checks["two_rdm_contraction_error"])
        <= 1.0e-8,
        "active_space_rdm_hermiticity": float(rdm_checks["two_rdm_hermiticity_error"])
        <= 1.0e-8,
        "active_space_rdm_energy": rdm_energy_error <= 1.0e-8,
    }
    gates.update(dict(additional or {}))
    return gates


def candidate_gates(
    *,
    energy: float,
    published_energy: float = PUBLISHED_CAS6_ENERGY_HARTREE,
    energy_tolerance: float,
    solver: Mapping[str, Any],
    rdm_checks: Mapping[str, float],
    rdm_energy_error: float,
    additional: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """Return the physical gates plus the historical paper-energy gate."""

    gates = physical_candidate_gates(
        energy=energy,
        solver=solver,
        rdm_checks=rdm_checks,
        rdm_energy_error=rdm_energy_error,
        additional=additional,
    )
    gates["published_active_space_energy"] = (
        math.isfinite(float(published_energy))
        and math.isfinite(float(energy_tolerance))
        and energy_tolerance > 0.0
        and abs(energy - published_energy) <= energy_tolerance
    )
    return gates


def write_candidate_artifacts(
    *,
    output_directory: Path,
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    active_indices: np.ndarray,
    reference_determinant: np.ndarray,
    energy: float,
    ci: np.ndarray,
    gamma: np.ndarray,
    gamma2: np.ndarray,
    cumulant: np.ndarray,
    extra_arrays: Mapping[str, np.ndarray],
    parent_orbital_count: int = PARENT_NORB,
) -> dict[str, dict[str, Any]]:
    # Imports stay here so parsing and unit testing the acceptance scaffold does
    # not require PennyLane to build any Hamiltonian.
    from benchmark_qc.integral_dataset import (
        SpatialIntegralArchive,
        jordan_wigner_terms_from_integrals,
        write_legacy_hamiltonian_npz,
    )

    source_path = output_directory / "source_integrals.npz"
    np.savez_compressed(
        source_path,
        geometry_angstrom=np.empty((0, 3), dtype=float),
        one_body_integrals=np.asarray(h1, dtype=float),
        two_body_integrals=np.asarray(eri, dtype=float),
        core_constant=np.asarray([ecore], dtype=float),
        # Portable archives index only the retained active frame. Parent-frame
        # indices remain available in the explicit provenance array below.
        active_mo_indices=np.arange(len(active_indices), dtype=np.int64),
        parent_active_mo_indices=np.asarray(active_indices, dtype=np.int64),
        parent_active_spatial_orbital_count=np.asarray(
            parent_orbital_count, dtype=np.int64
        ),
        reference_determinant=np.asarray(reference_determinant, dtype=bool),
        fci_coefficients=np.asarray(ci),
        one_rdm_spinorbital=np.asarray(gamma),
        two_rdm_spinorbital=np.asarray(gamma2),
        two_body_cumulant_spinorbital=np.asarray(cumulant),
        **{key: np.asarray(value) for key, value in extra_arrays.items()},
    )
    archive = SpatialIntegralArchive(
        geometry_angstrom=np.empty((0, 3), dtype=float),
        one_body_integrals=np.asarray(h1, dtype=float),
        two_body_integrals=np.asarray(eri, dtype=float),
        core_constant=float(ecore),
        active_mo_indices=np.arange(len(active_indices), dtype=np.int64),
        reference_determinant=np.asarray(reference_determinant, dtype=bool),
    )
    terms = jordan_wigner_terms_from_integrals(archive, cutoff=1.0e-14)
    hamiltonian_path = output_directory / "Fe2S2_H.npz"
    # The parent FCIDUMP does not encode coordinates.  Label 0.0 is therefore
    # an archive point identifier, not a bond length.
    write_legacy_hamiltonian_npz(
        hamiltonian_path,
        labels=[0.0],
        hamiltonian_terms=[terms],
        casci_energies=[energy],
    )
    return {
        "source_integrals": {
            "path": source_path.name,
            "sha256": sha256_file(source_path),
        },
        "hamiltonian": {
            "path": hamiltonian_path.name,
            "sha256": sha256_file(hamiltonian_path),
            "pauli_term_count": int(len(terms)),
        },
    }


def runtime_record(started: float, site_record: Mapping[str, Any]) -> dict[str, Any]:
    import pyscf

    return {
        **dict(site_record),
        "username": os.environ.get("USER"),
        "numpy": np.__version__,
        "pyscf": pyscf.__version__,
        "wall_seconds": time.time() - started,
    }
