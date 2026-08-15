"""Recover normalized spatial integrals for historical Hamiltonian-only systems.

Some of the historical BenchmarkQC point archives retained only a pickled
PennyLane Jordan--Wigner Hamiltonian.  This tool solves the inverse linear
map from that Hamiltonian to a real, spatial-orbital one-/two-electron
integral representation with the usual chemist-ERI symmetries.  The result is
an exact operator-equivalent representation for the saved Hamiltonian, not a
claim that the original AO/MO coefficient frame is recoverable from the JW
operator alone.

The generated archives are deliberately marked as explicit coefficient-space
archives and carry an identity active-frame matrix.  They are suitable for
reconstructing the published qubit Hamiltonian and for algorithms that need
numeric one-/two-electron coefficients.  Orbital-frame-sensitive classical
quantities should use the scalar reference records and provenance metadata,
not interpret the recovered frame as the original chemistry calculation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_qc.hamiltonian_test import load_hamiltonian_npz
from benchmark_qc.integral_dataset import (
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    max_pauli_coefficient_difference,
    pauli_coefficient_map,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "datasets" / "catalog.json"
COEFFICIENT_SOURCE = (
    "Jordan-Wigner-equivalent active-space frame; original orbital coefficients unavailable"
)
RECONSTRUCTION_TOLERANCE = 1.0e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pauli_map_from_operator(operator: object) -> dict[tuple[tuple[int, str], ...], complex]:
    import pennylane as qml

    result: dict[tuple[tuple[int, str], ...], complex] = {}
    for word, coefficient in qml.pauli.pauli_sentence(operator).items():
        key = tuple(sorted((int(wire), str(pauli)) for wire, pauli in word.items()))
        result[key] = result.get(key, 0.0j) + complex(coefficient)
    return {key: value for key, value in result.items() if abs(value) > 1.0e-15}


def _eri_orbit(index_tuple: tuple[int, int, int, int]) -> frozenset[tuple[int, int, int, int]]:
    p, q, r, s = index_tuple
    orbit: set[tuple[int, int, int, int]] = set()
    for left in ((p, q), (q, p)):
        for right in ((r, s), (s, r)):
            orbit.add((*left, *right))
            orbit.add((*right, *left))
    return frozenset(orbit)


def _forward_integrals(
    core_constant: float,
    one_body: np.ndarray,
    two_body: np.ndarray,
) -> dict[tuple[tuple[int, str], ...], complex]:
    import pennylane as qml

    fermionic = qml.qchem.fermionic_observable(
        np.asarray([core_constant], dtype=float),
        np.asarray(one_body, dtype=float),
        np.swapaxes(np.asarray(two_body, dtype=float), 1, 3),
        cutoff=1.0e-20,
    )
    return _pauli_map_from_operator(qml.jordan_wigner(fermionic))


def _basis_system(n_orbitals: int) -> tuple[np.ndarray, list[tuple[str, Any]]]:
    """Build the JW image of every independent spatial integral coefficient."""

    if n_orbitals < 1:
        raise ValueError("at least one spatial orbital is required")

    specs: list[tuple[str, Any]] = [("constant", None)]
    for p in range(n_orbitals):
        for q in range(p, n_orbitals):
            specs.append(("one", (p, q)))

    seen: set[frozenset[tuple[int, int, int, int]]] = set()
    for p in range(n_orbitals):
        for q in range(p, n_orbitals):
            for r in range(n_orbitals):
                for s in range(r, n_orbitals):
                    orbit = _eri_orbit((p, q, r, s))
                    if orbit in seen:
                        continue
                    seen.add(orbit)
                    specs.append(("two", tuple(sorted(orbit))))

    columns: list[dict[tuple[tuple[int, str], ...], complex]] = []
    for kind, payload in specs:
        one = np.zeros((n_orbitals, n_orbitals), dtype=float)
        two = np.zeros((n_orbitals,) * 4, dtype=float)
        constant = 0.0
        if kind == "constant":
            constant = 1.0
        elif kind == "one":
            p, q = payload
            one[p, q] = 1.0
            one[q, p] = 1.0
            if p == q:
                one[p, q] = 1.0
        else:
            value = 1.0
            for index_tuple in payload:
                two[index_tuple] = value
        columns.append(_forward_integrals(constant, one, two))

    keys = sorted(set().union(*(column.keys() for column in columns)))
    key_index = {key: index for index, key in enumerate(keys)}
    matrix = np.zeros((len(keys), len(columns)), dtype=complex)
    for column_index, column in enumerate(columns):
        for key, value in column.items():
            matrix[key_index[key], column_index] = value
    return matrix, specs


def _solve_integrals(
    hamiltonian_terms: Iterable[object],
    n_orbitals: int,
    basis_cache: dict[int, tuple[np.ndarray, list[tuple[str, Any]], list[tuple[tuple[int, str], ...]]]],
) -> tuple[float, np.ndarray, np.ndarray, float, int, float]:
    if n_orbitals not in basis_cache:
        matrix, specs = _basis_system(n_orbitals)
        # Rebuild the key order from the matrix-generating basis maps.  The
        # target is aligned below by using the same Pauli-word universe.
        constant = np.zeros((n_orbitals, n_orbitals), dtype=float)
        two = np.zeros((n_orbitals,) * 4, dtype=float)
        key_set: set[tuple[tuple[int, str], ...]] = set()
        for kind, payload in specs:
            one = np.zeros_like(constant)
            candidate_two = np.zeros_like(two)
            candidate_constant = 1.0 if kind == "constant" else 0.0
            if kind == "one":
                p, q = payload
                one[p, q] = 1.0
                one[q, p] = 1.0
                if p == q:
                    one[p, q] = 1.0
            elif kind == "two":
                for index_tuple in payload:
                    candidate_two[index_tuple] = 1.0
            key_set.update(_forward_integrals(candidate_constant, one, candidate_two))
        basis_cache[n_orbitals] = (matrix, specs, sorted(key_set))

    matrix, specs, keys = basis_cache[n_orbitals]
    # Historical PennyLane archives can retain round-off-only mixed X/Y terms
    # at ~1e-15.  A real spatial integral tensor has no independent
    # coefficient for those terms; discard only this numerical noise before
    # solving, while the post-write operator check still sees the raw archive.
    target_map = {
        key: value
        for key, value in pauli_coefficient_map(hamiltonian_terms).items()
        if abs(value) > 1.0e-12
    }
    key_index = {key: index for index, key in enumerate(keys)}
    if unknown := set(target_map).difference(key_index):
        raise RuntimeError(
            f"target Hamiltonian contains Pauli words outside the one-/two-body basis: {sorted(unknown)[:3]}"
        )
    target = np.zeros(len(keys), dtype=complex)
    for key, value in target_map.items():
        target[key_index[key]] = value
    if matrix.shape[0] != len(keys):
        raise RuntimeError("internal basis-key construction mismatch")

    solution, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=1.0e-12)
    residual = float(np.max(np.abs(matrix @ solution - target)))
    if residual > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            f"inverse JW reconstruction residual {residual:.3e} exceeds "
            f"{RECONSTRUCTION_TOLERANCE:.3e}"
        )

    core_constant = float(solution[0].real)
    one_body = np.zeros((n_orbitals, n_orbitals), dtype=float)
    cursor = 1
    for kind, payload in specs[1:]:
        if kind != "one":
            break
        p, q = payload
        value = float(solution[cursor].real)
        one_body[p, q] = value
        one_body[q, p] = value
        cursor += 1

    two_body = np.zeros((n_orbitals,) * 4, dtype=float)
    for kind, payload in specs[cursor:]:
        if kind != "two":
            raise RuntimeError("internal basis specification ordering mismatch")
        value = float(solution[cursor].real)
        for index_tuple in payload:
            two_body[index_tuple] = value
        cursor += 1

    smallest_kept_singular_value = float(singular_values[-1]) if len(singular_values) else 0.0
    return (
        core_constant,
        one_body,
        two_body,
        residual,
        int(rank),
        smallest_kept_singular_value,
    )


def _geometry(system: str, bond_length: float) -> np.ndarray:
    if system == "FeS":
        return np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]], dtype=float)
    return np.asarray(
        [[0.0, 0.0, -bond_length / 2.0], [0.0, 0.0, bond_length / 2.0]],
        dtype=float,
    )


def _reference_determinant(n_orbitals: int, electrons: int, spin_2s: int) -> np.ndarray:
    if (electrons + spin_2s) % 2:
        raise ValueError("electron count and spin must have matching parity")
    n_alpha = (electrons + spin_2s) // 2
    n_beta = (electrons - spin_2s) // 2
    if min(n_alpha, n_beta) < 0 or max(n_alpha, n_beta) > n_orbitals:
        raise ValueError("active electron/spin values do not fit the active space")
    reference = np.zeros(2 * n_orbitals, dtype=bool)
    reference[0 : 2 * n_alpha : 2] = True
    reference[1 : 2 * n_beta : 2] = True
    return reference


def _recover_entry(
    entry: dict[str, Any],
    *,
    basis_cache: dict[int, tuple[np.ndarray, list[tuple[str, Any]], list[tuple[tuple[int, str], ...]]]],
    write: bool,
) -> dict[str, Any]:
    hamiltonian_path = ROOT / entry["data_path"]
    metadata_path = ROOT / entry["metadata_path"]
    data = load_hamiltonian_npz(hamiltonian_path)
    if len(data.hs) != 1:
        raise ValueError(f"{entry['id']}: expected one point, found {len(data.hs)}")
    hamiltonian_terms = data.hs[0]
    n_qubits = int(entry["qubits"])
    if n_qubits % 2 or n_qubits != 2 * int(entry["active_spatial_orbitals"]):
        raise ValueError(f"{entry['id']}: inconsistent qubit/orbital metadata")
    n_orbitals = n_qubits // 2
    core, one_body, two_body, residual, rank, singular_value = _solve_integrals(
        hamiltonian_terms, n_orbitals, basis_cache
    )
    bond_length = float(entry["point_label"])
    reference = _reference_determinant(
        n_orbitals, int(entry["active_electrons"]), int(entry["spin_2S"])
    )
    occupations = reference[0::2].astype(float) + reference[1::2].astype(float)
    case_root = hamiltonian_path.parent
    integral_path = case_root / "inputs" / "source_integrals.npz"
    if write:
        integral_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            integral_path,
            geometry_angstrom=_geometry(entry["system"], bond_length),
            one_body_integrals=one_body,
            two_body_integrals=two_body,
            core_constant=np.asarray([core], dtype=float),
            active_mo_indices=np.empty(0, dtype=np.int64),
            reference_determinant=reference,
            selected_mp2_natural_indices=np.empty(0, dtype=np.int64),
            selected_mp2_natural_coeff=np.empty((n_orbitals, 0), dtype=float),
            pre_hf_active_mo_coeff=np.eye(n_orbitals, dtype=float),
            pre_hf_active_coefficient_source=np.asarray(COEFFICIENT_SOURCE),
            pre_hf_projected_mp2_natural_occupation_data=occupations,
        )
        archive = load_spatial_integral_archive(integral_path)
        regenerated_terms = jordan_wigner_terms_from_integrals(archive, cutoff=1.0e-14)
        regenerated_error = max_pauli_coefficient_difference(
            hamiltonian_terms, regenerated_terms
        )
        if regenerated_error > RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(
                f"{entry['id']}: saved archive regenerates with error {regenerated_error:.3e}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        relative_integral = integral_path.relative_to(ROOT).as_posix()
        digest = sha256_file(integral_path)
        metadata["integrals"] = {
            "path": relative_integral,
            "sha256": digest,
            "format": "pickle-free normalized spatial active-space archive",
            "frame_status": "Jordan-Wigner-equivalent; original orbital coefficient frame unavailable",
        }
        metadata.setdefault("provenance", {})["integral_reconstruction"] = {
            "tool": "tools/recover_missing_integrals.py",
            "method": "minimum-norm inverse Jordan-Wigner reconstruction over real spatial one-/two-body coefficients",
            "operator_equivalence_max_pauli_error": regenerated_error,
            "linear_system_rank": rank,
            "linear_system_smallest_retained_singular_value": singular_value,
            "warning": "This archive reproduces the saved JW Hamiltonian but does not recover the original AO/MO orbital frame.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entry["integral_data_path"] = relative_integral
        entry["integral_sha256"] = digest
    else:
        regenerated_error = residual
    return {
        "id": entry["id"],
        "n_spatial_orbitals": n_orbitals,
        "linear_system_rank": rank,
        "linear_system_smallest_retained_singular_value": singular_value,
        "inverse_residual_max": residual,
        "operator_equivalence_max_pauli_error": regenerated_error,
        "integral_data_path": entry.get("integral_data_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check existing recovered archives without writing")
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in catalog["datasets"]
        if entry.get("system") != "MolVQE21" and not entry.get("integral_data_path")
    ]
    if not entries:
        print("no missing BenchmarkQC integral archives")
        return

    basis_cache: dict[int, tuple[np.ndarray, list[tuple[str, Any]], list[tuple[tuple[int, str], ...]]]] = {}
    report = []
    for index, entry in enumerate(entries, start=1):
        result = _recover_entry(entry, basis_cache=basis_cache, write=not args.check)
        report.append(result)
        print(
            f"[{index}/{len(entries)}] {entry['id']}: "
            f"residual={result['inverse_residual_max']:.3e}, "
            f"operator_error={result['operator_equivalence_max_pauli_error']:.3e}"
        )

    if not args.check:
        CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"processed {len(report)} missing BenchmarkQC integral archives")


if __name__ == "__main__":
    main()
