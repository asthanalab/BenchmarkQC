"""Build portable MolVQE-21 Benchmark-QC archives from benchmarkkrylov.

The benchmarkkrylov repository stores the corrected MolVQE references as
numeric active-space integral caches.  This script normalizes those caches to
the Benchmark-QC integral contract and emits the historical three-array Hamiltonian
NPZ contract used by the rest of this repository.

The source checkout is passed explicitly so the generated files remain
portable and do not contain workstation-specific paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from benchmark_qc.integral_dataset import (
        jordan_wigner_terms_from_integrals,
        load_spatial_integral_archive,
    )
except ModuleNotFoundError:  # Allow direct execution from the dataset directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from benchmark_qc.integral_dataset import (
        jordan_wigner_terms_from_integrals,
        load_spatial_integral_archive,
    )


SOURCE_COMMIT = "2f1faef5589dd276393be4cb93860524f4ff790a"
SOURCE_REPOSITORY = "https://github.com/asthanaa/benchmarkkrylov"
ANGSTROM_PER_BOHR = 0.529177210903


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(f"expected a scalar array, got shape {value.shape}")
    return value.reshape(-1)[0]


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if not value else float(value)


def _source_case(source_root: Path, row: dict[str, str]) -> dict[str, Any]:
    source_path = source_root / row["source_json_path"]
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    meta = payload["meta"]
    molecule = meta["molecule"]
    casscf = meta["casscf"]
    atoms = molecule["atoms"]
    geometry_bohr = [[float(value) for value in atom["xyz"]] for atom in atoms]
    return {
        "case_id": row["case_id"],
        "source_name": row["source_name"],
        "source_json_path": row["source_json_path"],
        "source_json_sha256": _sha256(source_path),
        "symbols": [str(atom["element"]) for atom in atoms],
        "geometry_bohr": geometry_bohr,
        "geometry_angstrom": [
            [value * ANGSTROM_PER_BOHR for value in xyz]
            for xyz in geometry_bohr
        ],
        "basis": str(molecule["basis"]),
        "charge": int(molecule["charge"]),
        "spin": int(molecule["spin"]),
        "active_electrons": int(casscf["nelec"]),
        "active_orbitals": int(casscf["ncas"]),
        "source_casscf_energy_hartree": float(casscf["energy"]),
        "source_casscf_converged": bool(casscf["converged"]),
        "source_casscf_natorb": bool(casscf["natorb"]),
        "renyi_0p25_nats": _optional_float(row, "h0.25"),
    }


def _reference_determinant(ncas: int, nelecas: np.ndarray) -> np.ndarray:
    values = np.asarray(nelecas, dtype=int).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"nelecas must contain alpha and beta counts, got {values}")
    n_alpha, n_beta = (int(values[0]), int(values[1]))
    if n_alpha < 0 or n_beta < 0 or n_alpha > ncas or n_beta > ncas:
        raise ValueError(f"invalid active-space occupation ({n_alpha}, {n_beta}) for ncas={ncas}")
    determinant = np.zeros(2 * ncas, dtype=bool)
    determinant[0 : 2 * n_alpha : 2] = True
    determinant[1 : 2 * n_beta : 2] = True
    return determinant


def _write_case(
    *,
    source_root: Path,
    output_root: Path,
    source_case: dict[str, Any],
    cache_row: dict[str, str],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    case_id = source_case["case_id"]
    source_cache_path = source_root / cache_row["source_cache_path"]
    with np.load(source_cache_path, allow_pickle=False) as cache:
        ncas = int(_scalar(cache["ncas"]))
        nelecas = np.asarray(cache["nelecas"], dtype=int)
        reference_energy = float(_scalar(cache["reference_energy"]))
        effective_active_electrons = int(_scalar(cache["effective_active_electrons"]))
        ecore = float(_scalar(cache["ecore"]))
        one_body = np.asarray(cache["h1ecas"], dtype=float)
        source_two_mo = np.asarray(cache["two_mo"], dtype=float)
        reference_converged = bool(_scalar(cache["reference_converged"]))
        hamiltonian_source = str(_scalar(cache["hamiltonian_source"]))
        mo_occ = np.asarray(cache["mo_occ"], dtype=float)

    if ncas != source_case["active_orbitals"]:
        raise ValueError(f"{case_id}: cache ncas={ncas} disagrees with source manifest")
    if effective_active_electrons != source_case["active_electrons"]:
        raise ValueError(
            f"{case_id}: cache active electron count {effective_active_electrons} "
            f"disagrees with source manifest {source_case['active_electrons']}"
        )
    if one_body.shape != (ncas, ncas) or source_two_mo.shape != (ncas,) * 4:
        raise ValueError(
            f"{case_id}: unexpected integral shapes one={one_body.shape}, two={source_two_mo.shape}"
        )
    if mo_occ.ndim != 1 or not np.all(np.isfinite(mo_occ)):
        raise ValueError(f"{case_id}: invalid full-space orbital occupation data")

    active_orbital_indices = [] if override is None else [int(v) for v in override["active_orbital_indices"]]
    active_orbital_index_base = 0 if override is None else int(override.get("active_orbital_index_base", 0))
    selection_method = (
        "source cached CASSCF natural-orbital active space; global MO indices not exported"
        if override is None
        else str(override.get("selection_method", "pyscf_canonical_rohf_mo_indices"))
    )

    system_dir = output_root / "systems" / case_id
    integrals_dir = system_dir / "inputs"
    integrals_path = integrals_dir / "source_integrals.npz"
    hamiltonian_path = system_dir / "hamiltonian.npz"
    integrals_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        integrals_path,
        geometry_angstrom=np.asarray(source_case["geometry_angstrom"], dtype=float),
        one_body_integrals=one_body,
        # benchmarkkrylov stores the physicist-like source tensor.  The
        # Benchmark-QC archive contract stores chemist ERIs, so normalize once
        # here; the shared Jordan-Wigner converter swaps this axis back when it
        # builds the fermionic observable.
        two_body_integrals=np.swapaxes(source_two_mo, 1, 3),
        core_constant=np.asarray([ecore], dtype=float),
        # These are local active-space positions, not fabricated global MO indices.
        active_mo_indices=np.arange(ncas, dtype=np.int64),
        reference_determinant=_reference_determinant(ncas, nelecas),
        active_orbital_indices=np.asarray(active_orbital_indices, dtype=np.int64),
        active_orbital_index_base=np.asarray([active_orbital_index_base], dtype=np.int64),
        active_orbital_selection_method=np.asarray(selection_method),
        effective_active_electrons=np.asarray([effective_active_electrons], dtype=np.int64),
        reference_energy=np.asarray([reference_energy], dtype=float),
        reference_converged=np.asarray([reference_converged], dtype=bool),
        hamiltonian_source=np.asarray(hamiltonian_source),
        source_cache_sha256=np.asarray(_sha256(source_cache_path)),
    )

    archive = load_spatial_integral_archive(integrals_path)
    terms = jordan_wigner_terms_from_integrals(archive, cutoff=1e-20)
    labels = np.asarray([0.0], dtype=object)
    hamiltonians = np.empty(1, dtype=object)
    hamiltonians[0] = terms
    np.savez_compressed(
        hamiltonian_path,
        labels=labels,
        Hs=hamiltonians,
        casci_energies=np.asarray([reference_energy], dtype=np.float64),
    )

    return {
        **source_case,
        "source_cache_path": cache_row["source_cache_path"],
        "source_cache_sha256": _sha256(source_cache_path),
        "source_cache_reference_energy_hartree": reference_energy,
        "reference_converged": reference_converged,
        "hamiltonian_source": hamiltonian_source,
        "effective_active_electrons": effective_active_electrons,
        "active_orbital_indices": active_orbital_indices,
        "active_orbital_index_base": active_orbital_index_base,
        "active_orbital_selection_method": selection_method,
        "integrals_path": str(integrals_path.relative_to(output_root)),
        "integrals_sha256": _sha256(integrals_path),
        "hamiltonian_path": str(hamiltonian_path.relative_to(output_root)),
        "hamiltonian_sha256": _sha256(hamiltonian_path),
        "metadata_path": str((system_dir / "metadata.json").relative_to(output_root)),
        "pauli_term_count": len(terms),
        "qubits": 2 * ncas,
    }


def _write_system_metadata(*, output_root: Path, record: dict[str, Any]) -> None:
    """Write one system record using the shared Benchmark-QC metadata shape."""

    case_id = record["case_id"]
    system_dir = output_root / "systems" / case_id
    relative_hamiltonian = record["hamiltonian_path"]
    relative_integrals = record["integrals_path"]
    renyi = record.get("renyi_0p25_nats")
    metadata = {
        "schema": "benchmark-qc.system-metadata.v1",
        "system": "MolVQE21",
        "system_id": case_id,
        "active_space_variant": None,
        "active_space": {
            "active_electrons": record["active_electrons"],
            "active_spatial_orbitals": record["active_orbitals"],
            "qubits": record["qubits"],
            "spin_2S": record["spin"],
        },
        "basis": record["basis"],
        "charge": record["charge"],
        "geometry": {
            "coordinates_bohr": record["geometry_bohr"],
            "coordinates_angstrom": record["geometry_angstrom"],
            "point_label": 0.0,
            "point_name": "reference",
            "symbols": record["symbols"],
        },
        "hamiltonian": {
            "format": "labels, Hs, casci_energies; one point",
            "path": str(Path("datasets/molvqe21") / relative_hamiltonian),
            "pauli_term_count": record["pauli_term_count"],
            "sha256": record["hamiltonian_sha256"],
        },
        "integrals": {
            "format": "pickle-free normalized spatial active-space archive",
            "path": str(Path("datasets/molvqe21") / relative_integrals),
            "sha256": record["integrals_sha256"],
        },
        "provenance": {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_json_path": record["source_json_path"],
            "source_json_sha256": record["source_json_sha256"],
            "source_cache_path": record["source_cache_path"],
            "source_cache_sha256": record["source_cache_sha256"],
            "source_cache_reference_energy_hartree": record[
                "source_cache_reference_energy_hartree"
            ],
        },
        "reference_energy_hartree": record["source_cache_reference_energy_hartree"],
        "reference_results": {
            "schema": "benchmark-qc.scalar-reference-results.v1",
            "casci_energy_hartree": record["source_cache_reference_energy_hartree"],
            "cisd_energy_hartree": None,
            "cisd_status": "not-provided-by-source",
            "ccsd_energy_hartree": None,
            "ccsd_status": "not-provided-by-source",
            "renyi_0p25_nats": renyi,
            "cumulant_C2": None,
            "source": {
                "kind": "MolVQE-21 corrected cache",
                "system_id": case_id,
            },
        },
        "source_active_space": {
            "active_orbital_indices": record["active_orbital_indices"],
            "active_orbital_index_base": record["active_orbital_index_base"],
            "active_orbital_selection_method": record[
                "active_orbital_selection_method"
            ],
            "effective_active_electrons": record["effective_active_electrons"],
            "source_casscf_converged": record["source_casscf_converged"],
            "source_casscf_energy_hartree": record["source_casscf_energy_hartree"],
            "source_casscf_natorb": record["source_casscf_natorb"],
        },
        "source_metadata_path": None,
        "si_validation": None,
    }
    (system_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(source_root: Path, output_root: Path) -> None:
    source_manifest_path = source_root / "manifests/molvqe21_fig1b_manifest.csv"
    corrected_manifest_path = source_root / "hamiltonian_cache/correct_27_systems/manifest.csv"
    overrides_path = source_root / "manifests/molvqe21_active_orbital_overrides.json"
    source_rows = _read_csv(source_manifest_path)
    corrected_rows = _read_csv(corrected_manifest_path)
    source_by_case = {row["case_id"]: row for row in source_rows}
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "active_orbital_overrides.json").write_text(
        json.dumps(overrides, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    for cache_row in corrected_rows:
        case_id = cache_row["case_id"]
        if case_id not in source_by_case:
            raise ValueError(f"corrected cache case {case_id!r} is missing from MolVQE manifest")
        source_case = _source_case(source_root, source_by_case[case_id])
        records.append(
            _write_case(
                source_root=source_root,
                output_root=output_root,
                source_case=source_case,
                cache_row=cache_row,
                override=overrides.get(case_id),
            )
        )

    for record in records:
        _write_system_metadata(output_root=output_root, record=record)

    # Keep only the source-manifest rows that have corrected benchmark caches.
    # The two incomplete source records are intentionally omitted from this
    # package rather than carried as non-runnable catalog entries.
    ready_case_ids = {record["case_id"] for record in records}
    filtered_source_rows = [row for row in source_rows if row["case_id"] in ready_case_ids]
    with (output_root / "source_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(filtered_source_rows)

    manifest_fields = [
        "case_id",
        "hamiltonian_path",
        "hamiltonian_sha256",
        "integrals_path",
        "integrals_sha256",
        "metadata_path",
        "pauli_term_count",
        "source_json_path",
        "source_cache_path",
        "source_cache_sha256",
        "symbols",
        "basis",
        "charge",
        "spin",
        "active_electrons",
        "active_orbitals",
        "qubits",
        "source_casscf_energy_hartree",
        "source_cache_reference_energy_hartree",
        "reference_converged",
        "active_orbital_indices",
        "active_orbital_selection_method",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in manifest_fields}
            row["symbols"] = json.dumps(record["symbols"], separators=(",", ":"))
            row["active_orbital_indices"] = json.dumps(
                record["active_orbital_indices"], separators=(",", ":")
            )
            writer.writerow(row)

    metadata = {
        "schema": "benchmark-qc.dataset-metadata.v1",
        "systems_root": "systems",
        "system": {
            "name": "MolVQE-21",
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_manifest": "source_manifest.csv",
            "corrected_cache_manifest": "manifest.csv",
            "active_orbital_overrides": "active_orbital_overrides.json",
        },
        "source_case_count": len(filtered_source_rows),
        "benchmark_ready_case_count": len(records),
        "cases": {
            record["case_id"]: {
                key: value
                for key, value in record.items()
                if key not in {"geometry_bohr", "geometry_angstrom"}
            }
            for record in records
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
