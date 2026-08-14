"""Load the corrected MolVQE-21 Benchmark-QC datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .hamiltonian_test import NPZData, load_hamiltonian_npz
from .integral_dataset import SpatialIntegralArchive, load_spatial_integral_archive
from .paths import MOLVQE21_ROOT


ROOT = MOLVQE21_ROOT
MANIFEST_PATH = ROOT / "manifest.csv"


@dataclass(frozen=True)
class MolVQE21Case:
    """Portable metadata for one corrected MolVQE-21 reference."""

    case_id: str
    hamiltonian_path: Path
    integrals_path: Path
    active_electrons: int
    active_orbitals: int
    qubits: int
    charge: int
    spin: int
    basis: str
    reference_energy_hartree: float
    reference_converged: bool


def _load_cases() -> dict[str, MolVQE21Case]:
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["case_id"]: MolVQE21Case(
            case_id=row["case_id"],
            hamiltonian_path=ROOT / row["hamiltonian_path"],
            integrals_path=ROOT / row["integrals_path"],
            active_electrons=int(row["active_electrons"]),
            active_orbitals=int(row["active_orbitals"]),
            qubits=int(row["qubits"]),
            charge=int(row["charge"]),
            spin=int(row["spin"]),
            basis=row["basis"],
            reference_energy_hartree=float(row["source_cache_reference_energy_hartree"]),
            reference_converged=row["reference_converged"].lower() == "true",
        )
        for row in rows
    }


def list_cases() -> tuple[MolVQE21Case, ...]:
    """Return all benchmark-ready corrected MolVQE-21 cases."""

    return tuple(_load_cases().values())


def get_case(case_id: str) -> MolVQE21Case:
    """Return metadata for one case by its stable manifest identifier."""

    try:
        return _load_cases()[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown MolVQE-21 case: {case_id!r}") from exc


def load_hamiltonian(case_id: str) -> NPZData:
    """Load one corrected case using the standard Benchmark-QC NPZ contract."""

    case = get_case(case_id)
    return load_hamiltonian_npz(str(case.hamiltonian_path))


def load_source_integrals(case_id: str) -> SpatialIntegralArchive:
    """Load one normalized, pickle-free corrected integral archive."""

    case = get_case(case_id)
    return load_spatial_integral_archive(case.integrals_path)
