"""Materialize common scalar reference results for every BenchmarkQC point.

The SI Table I rows are copied into the per-system metadata using the table's
published values.  Historical N2, FeS, and U2 points that are not represented in
that table are evaluated from their checked-in normalized integral archives.
The latter values are explicitly marked as calculations in the recovered
Jordan--Wigner-equivalent frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_qc.integral_dataset import load_spatial_integral_archive


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "datasets" / "catalog.json"
RESULTS_PATH = ROOT / "datasets" / "benchmarkQC" / "reference_results.json"
DEFAULT_TABLE = Path(
    "/Users/ayushasthana/Library/CloudStorage/OneDrive2-NorthDakotaUniversitySystem/"
    "code/qcomputing/U2response/DD-ART-06-2026-000368/"
    "submission_bundle_Fe2S2_candidate_v2_2026-08-12/05_supporting_data/"
    "unified_benchmark_rows.json"
)
FE2S2_CONTROL_IDS = {
    "fe2s2_chan30e20o_vi66_cas4e4o_historical": "fe2s2_chan30e20o_historical_rhf_cas4e4o_point0",
    "fe2s2_chan30e20o_vi66_cas6e6o_historical": "fe2s2_chan30e20o_historical_rhf_cas6e6o_point0",
    "fe2s2_chan30e20o_vi66_cas8e6o_historical": "fe2s2_chan30e20o_historical_rhf_cas8e6o_point0",
    "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o": "fe2s2_chan30e20o_vi66_rhf_nested_cas8e8o_point0",
}


def _spin_orbital_rdms(
    ci_vector: np.ndarray,
    *,
    n_orbitals: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pyscf import fci

    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = fci.direct_spin1.make_rdm12s(
        ci_vector, n_orbitals, nelec
    )
    nso = 2 * n_orbitals
    gamma = np.zeros((nso, nso), dtype=np.result_type(ci_vector, float))
    gamma[:n_orbitals, :n_orbitals] = dm1a.T
    gamma[n_orbitals:, n_orbitals:] = dm1b.T
    gamma2 = np.zeros((nso, nso, nso, nso), dtype=gamma.dtype)
    gamma2[:n_orbitals, :n_orbitals, :n_orbitals, :n_orbitals] = dm2aa.transpose(0, 2, 1, 3)
    gamma2[n_orbitals:, n_orbitals:, n_orbitals:, n_orbitals:] = dm2bb.transpose(0, 2, 1, 3)
    ab = dm2ab.transpose(0, 2, 1, 3)
    gamma2[:n_orbitals, n_orbitals:, :n_orbitals, n_orbitals:] = ab
    gamma2[n_orbitals:, :n_orbitals, :n_orbitals, n_orbitals:] = -ab.transpose(1, 0, 2, 3)
    gamma2[:n_orbitals, n_orbitals:, n_orbitals:, :n_orbitals] = -ab.transpose(0, 1, 3, 2)
    gamma2[n_orbitals:, :n_orbitals, n_orbitals:, :n_orbitals] = ab.transpose(1, 0, 3, 2)
    disconnected = np.einsum("pr,qs->pqrs", gamma, gamma) - np.einsum(
        "ps,qr->pqrs", gamma, gamma
    )
    cumulant = gamma2 - disconnected
    interleaved = np.asarray(
        [index for spatial in range(n_orbitals) for index in (spatial, n_orbitals + spatial)],
        dtype=int,
    )
    return (
        gamma[np.ix_(interleaved, interleaved)],
        gamma2[np.ix_(interleaved, interleaved, interleaved, interleaved)],
        cumulant[np.ix_(interleaved, interleaved, interleaved, interleaved)],
    )


def _descriptors(ci_vector: np.ndarray, cumulant: np.ndarray) -> dict[str, float]:
    probabilities = np.abs(np.asarray(ci_vector).ravel()) ** 2
    probabilities /= float(np.sum(probabilities))
    retained = probabilities[probabilities > 1.0e-15]
    retained /= float(np.sum(retained))
    h025 = math.log(float(np.sum(retained**0.25))) / 0.75
    return {
        "renyi_0p25_nats": float(h025),
        "cumulant_C2": float(0.25 * np.vdot(cumulant, cumulant).real),
    }


def _fci_state(archive, *, electrons: int, spin_2s: int) -> tuple[float, np.ndarray, dict[str, float]]:
    from pyscf import fci

    nelec = ((electrons + spin_2s) // 2, (electrons - spin_2s) // 2)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1.0e-12
    solver.max_cycle = 1000
    energy, ci_vector = solver.kernel(
        archive.one_body_integrals,
        archive.two_body_integrals,
        archive.n_spatial_orbitals,
        nelec,
        ecore=archive.core_constant,
    )
    _, _, cumulant = _spin_orbital_rdms(
        ci_vector,
        n_orbitals=archive.n_spatial_orbitals,
        nelec=nelec,
    )
    return float(energy), np.asarray(ci_vector), _descriptors(ci_vector, cumulant)


def _singlet_controls(archive, *, electrons: int) -> dict[str, Any]:
    from pyscf import ao2mo, cc, ci, gto, scf

    n = archive.n_spatial_orbitals
    molecule = gto.Mole()
    molecule.nelectron = electrons
    molecule.spin = 0
    molecule.nao = n
    molecule.incore_anyway = True
    molecule.verbose = 0
    molecule.build()
    mean_field = scf.RHF(molecule)
    mean_field.get_hcore = lambda *unused: archive.one_body_integrals
    mean_field.get_ovlp = lambda *unused: np.eye(n)
    mean_field._eri = ao2mo.restore(8, archive.two_body_integrals, n)
    mean_field.energy_nuc = lambda *unused: float(archive.core_constant)
    occupied = electrons // 2
    occupations = np.asarray([2.0] * occupied + [0.0] * (n - occupied))
    coefficients = np.eye(n)
    density = mean_field.make_rdm1(coefficients, occupations)
    fock = mean_field.get_fock(dm=density)
    occupied_energies, occupied_rotation = np.linalg.eigh(fock[:occupied, :occupied])
    virtual_energies, virtual_rotation = np.linalg.eigh(fock[occupied:, occupied:])
    coefficients[:occupied, :occupied] = occupied_rotation
    coefficients[occupied:, occupied:] = virtual_rotation
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
    return {
        "cisd_energy_hartree": float(cisd.e_tot),
        "ccsd_energy_hartree": float(coupled_cluster.e_tot),
        "cisd_status": "computed" if cisd.converged else "computed-not-converged",
        "ccsd_status": "computed" if coupled_cluster.converged else "computed-not-converged",
        "reference_determinant_energy_hartree": float(mean_field.e_tot),
        "reference_occupied_virtual_gradient_norm": gradient_norm,
    }


def _high_spin_controls(archive, *, electrons: int, spin_2s: int) -> dict[str, Any]:
    """Run unrestricted controls for the historical FeS spin-4 scan."""

    from pyscf import ao2mo, cc, ci, gto, scf

    n = archive.n_spatial_orbitals
    n_alpha = (electrons + spin_2s) // 2
    n_beta = (electrons - spin_2s) // 2
    h = archive.one_body_integrals
    eri = archive.two_body_integrals
    density_alpha = np.diag([1.0] * n_alpha + [0.0] * (n - n_alpha))
    density_beta = np.diag([1.0] * n_beta + [0.0] * (n - n_beta))
    density_total = density_alpha + density_beta
    coulomb = np.einsum("pqrs,rs->pq", eri, density_total)
    fock_alpha = h + coulomb - np.einsum("prqs,rs->pq", eri, density_alpha)
    fock_beta = h + coulomb - np.einsum("prqs,rs->pq", eri, density_beta)

    coefficients = []
    orbital_energies = []
    for fock, occupied_count in ((fock_alpha, n_alpha), (fock_beta, n_beta)):
        occupied_energies, occupied_rotation = np.linalg.eigh(fock[:occupied_count, :occupied_count])
        virtual_energies, virtual_rotation = np.linalg.eigh(fock[occupied_count:, occupied_count:])
        coefficient = np.eye(n)
        coefficient[:occupied_count, :occupied_count] = occupied_rotation
        coefficient[occupied_count:, occupied_count:] = virtual_rotation
        coefficients.append(coefficient)
        orbital_energies.append(np.concatenate((occupied_energies, virtual_energies)))

    molecule = gto.Mole()
    molecule.nelectron = electrons
    molecule.spin = spin_2s
    molecule.incore_anyway = True
    molecule.verbose = 0
    molecule.build()
    molecule._nao = n
    mean_field = scf.UHF(molecule)
    mean_field.get_hcore = lambda *unused: h
    mean_field.get_ovlp = lambda *unused: np.eye(n)
    mean_field._eri = ao2mo.restore(8, eri, n)
    mean_field.energy_nuc = lambda *unused: float(archive.core_constant)
    occupations_alpha = np.asarray([1.0] * n_alpha + [0.0] * (n - n_alpha))
    occupations_beta = np.asarray([1.0] * n_beta + [0.0] * (n - n_beta))
    mean_field.mo_coeff = tuple(coefficients)
    mean_field.mo_occ = (occupations_alpha, occupations_beta)
    mean_field.mo_energy = tuple(orbital_energies)
    mean_field.e_tot = float(
        archive.core_constant
        + np.einsum("pq,pq->", h, density_total)
        + 0.5 * np.einsum("pqrs,pq,rs->", eri, density_total, density_total)
        - 0.5 * np.einsum("pqrs,pr,qs->", eri, density_alpha, density_alpha)
        - 0.5 * np.einsum("pqrs,pr,qs->", eri, density_beta, density_beta)
    )
    mean_field.converged = True

    cisd = ci.CISD(mean_field)
    cisd.conv_tol = 1.0e-11
    cisd.max_cycle = 1000
    cisd.kernel()
    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.conv_tol = 1.0e-12
    coupled_cluster.conv_tol_normt = 1.0e-10
    coupled_cluster.max_cycle = 1000
    coupled_cluster.diis_space = 12
    coupled_cluster.kernel()
    return {
        "cisd_energy_hartree": float(cisd.e_tot),
        "ccsd_energy_hartree": float(coupled_cluster.e_tot),
        "cisd_status": "computed" if cisd.converged else "computed-not-converged",
        "ccsd_status": "computed" if coupled_cluster.converged else "computed-not-converged",
        "reference_determinant_energy_hartree": float(mean_field.e_tot),
    }


def _table_case_id(row: dict[str, Any]) -> str:
    if row["system_id"] in FE2S2_CONTROL_IDS:
        return FE2S2_CONTROL_IDS[row["system_id"]]
    system_id = row["system_id"]
    geometry_id = row["geometry_id"]
    geometry_suffix = geometry_id
    strip_geometry_role = (
        system_id.startswith("n2_ccpvdz_cas10e8o_")
        or system_id.endswith("_historical")
        or system_id == "c2_cas8e8o_augccpvtz"
    )
    if strip_geometry_role:
        for prefix in ("equilibrium_", "stretched_"):
            if geometry_suffix.startswith(prefix):
                geometry_suffix = geometry_suffix.removeprefix(prefix)
                break
    if system_id == "c2_cas8e8o_augccpvtz":
        if geometry_id == "equilibrium_r1p2430":
            return "c2_cas8e8o_augccpvtz_equilibrium_r1p2430"
        return f"c2_augccpvtz_cas8e8o_casci_natural_orbitals_{geometry_suffix}"
    if system_id == "n2_ccpvdz_cas10e8o_casci_natural":
        system_id = "n2_ccpvdz_cas10e8o_casci_natural_orbitals"
    elif system_id.endswith("_historical") and not system_id.startswith("n2_sto6g_"):
        system_id = system_id.removesuffix("_historical")
    return f"{system_id}_{geometry_suffix}"


def _table_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "benchmark-qc.scalar-reference-results.v1",
        "casci_energy_hartree": float(row["casci"]),
        "cisd_energy_hartree": None if row.get("cisd") is None else float(row["cisd"]),
        "ccsd_energy_hartree": None if row.get("ccsd") is None else float(row["ccsd"]),
        "cisd_status": "table-value" if row.get("cisd") is not None else "unavailable",
        "ccsd_status": str(row.get("ccsd_status", "table-value")),
        "renyi_0p25_nats": float(row["renyi_0p25_nats"]),
        "cumulant_C2": float(row["cumulant_C2"]),
        "source": {
            "kind": "SI Table I scalar record",
            "system_id": row["system_id"],
            "geometry_id": row["geometry_id"],
            "result_source": row.get("result_source"),
        },
    }


def _computed_record(entry: dict[str, Any]) -> dict[str, Any]:
    archive = load_spatial_integral_archive(ROOT / entry["integral_data_path"])
    electrons = int(entry["active_electrons"])
    spin_2s = int(entry["spin_2S"])
    casci, _, descriptors = _fci_state(
        archive,
        electrons=electrons,
        spin_2s=spin_2s,
    )
    if spin_2s == 0:
        controls = _singlet_controls(archive, electrons=electrons)
    else:
        controls = _high_spin_controls(
            archive,
            electrons=electrons,
            spin_2s=spin_2s,
        )
    stored = float(entry["reference_energy_hartree"])
    if abs(casci - stored) > 1.0e-7:
        raise RuntimeError(f"{entry['id']}: computed CASCI {casci:.15f} disagrees with stored {stored:.15f}")
    return {
        "schema": "benchmark-qc.scalar-reference-results.v1",
        "casci_energy_hartree": stored,
        "cisd_energy_hartree": controls["cisd_energy_hartree"],
        "ccsd_energy_hartree": controls["ccsd_energy_hartree"],
        "cisd_status": controls["cisd_status"],
        "ccsd_status": controls["ccsd_status"],
        "renyi_0p25_nats": descriptors["renyi_0p25_nats"],
        "cumulant_C2": descriptors["cumulant_C2"],
        "source": {
            "kind": "computed from checked-in normalized integral archive",
            "method": "PySCF direct-spin1 CASCI/FCI plus CISD and CCSD controls",
            "frame_note": "For historical N2, FeS, and U2 cases the integral archive is JW-equivalent; the values are frame-dependent controls, not claims about the unavailable original orbital frame.",
            "reference_determinant_energy_hartree": controls.get("reference_determinant_energy_hartree"),
            "reference_occupied_virtual_gradient_norm": controls.get("reference_occupied_virtual_gradient_norm"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-json", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--check", action="store_true", help="verify existing result blocks without rewriting")
    args = parser.parse_args()
    if not args.table_json.is_file():
        raise SystemExit(f"table JSON not found: {args.table_json}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in catalog["datasets"] if entry["system"] != "MolVQE21"}
    table_rows = json.loads(args.table_json.read_text(encoding="utf-8"))["rows"]
    table_records = {_table_case_id(row): _table_record(row) for row in table_rows}
    unknown = sorted(set(table_records).difference(entries))
    if unknown:
        raise SystemExit(f"table rows do not map to catalog IDs: {unknown}")

    results: dict[str, dict[str, Any]] = {}
    for case_id, entry in sorted(entries.items()):
        if case_id in table_records:
            record = table_records[case_id]
        else:
            print(f"computing scalar controls for {case_id}")
            record = _computed_record(entry)
        results[case_id] = record
        if not args.check:
            metadata_path = ROOT / entry["metadata_path"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["reference_results"] = record
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    if not args.check:
        RESULTS_PATH.write_text(
            json.dumps(
                {
                    "schema": "benchmark-qc.reference-results.v1",
                    "description": "Scalar CASCI, CISD, CCSD, Rényi-0.25, and two-body-cumulant records for every BenchmarkQC point-level system.",
                    "case_count": len(results),
                    "cases": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"materialized scalar results for {len(results)} BenchmarkQC systems")


if __name__ == "__main__":
    main()
