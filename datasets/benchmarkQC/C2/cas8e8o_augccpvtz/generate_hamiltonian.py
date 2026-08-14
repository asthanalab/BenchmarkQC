#!/usr/bin/env python3
"""Generate or verify the C2 CAS(8e,8o) CASCI-natural Hamiltonian."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_qc.hamiltonian_test import load_hamiltonian_npz  # noqa: E402
from benchmark_qc.integral_dataset import (  # noqa: E402
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
    max_pauli_coefficient_difference,
    sha256_file,
    write_legacy_hamiltonian_npz,
)


VARIANT = "casci_natural_orbitals"
GEOMETRY_ID = "stretched_r2p2000"
BOND_LENGTH_ANGSTROM = 2.2
CASCI_ENERGY_HARTREE = -75.41966083838146


def _build() -> tuple[list[float], list[np.ndarray], list[float]]:
    source_path = HERE / VARIANT / "inputs" / f"{GEOMETRY_ID}.npz"
    archive = load_spatial_integral_archive(source_path)
    if archive.n_spatial_orbitals != 8:
        raise ValueError(f"{source_path}: expected eight active spatial orbitals")
    if not np.array_equal(archive.active_mo_indices, np.arange(2, 10)):
        raise ValueError(f"{source_path}: expected source-subspace indices 2..9")
    expected_reference = np.asarray([True] * 8 + [False] * 8, dtype=bool)
    if not np.array_equal(archive.reference_determinant, expected_reference):
        raise ValueError(f"{source_path}: unexpected eight-electron reference determinant")
    measured_bond = float(
        np.linalg.norm(archive.geometry_angstrom[1] - archive.geometry_angstrom[0])
    )
    if abs(measured_bond - BOND_LENGTH_ANGSTROM) > 1e-10:
        raise ValueError(
            f"{source_path}: geometry gives R={measured_bond}, "
            f"expected {BOND_LENGTH_ANGSTROM} Angstrom"
        )
    terms = jordan_wigner_terms_from_integrals(archive, cutoff=1e-14)
    return [BOND_LENGTH_ANGSTROM], [terms], [CASCI_ENERGY_HARTREE]


def generate(*, output_root: Path, force: bool) -> None:
    labels, hamiltonians, energies = _build()
    output = output_root / VARIANT / "C2_PES_H.npz"
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new --output-root or pass --force"
        )
    write_legacy_hamiltonian_npz(
        output,
        labels=labels,
        hamiltonian_terms=hamiltonians,
        casci_energies=energies,
    )
    print(
        f"Wrote {output}; terms={len(hamiltonians[0])}; "
        f"sha256={sha256_file(output)}"
    )


def check(*, atol: float = 1e-12) -> None:
    metadata = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    variant_metadata = metadata["variants"][VARIANT]
    for source in variant_metadata["source_integral_archives"]:
        source_path = HERE / source["path"]
        actual = sha256_file(source_path)
        if actual != source["sha256"]:
            raise RuntimeError(
                f"Source checksum mismatch for {source_path}: "
                f"expected {source['sha256']}, got {actual}"
            )

    output = HERE / variant_metadata["output"]["path"]
    output_hash = sha256_file(output)
    if output_hash != variant_metadata["output"]["sha256"]:
        raise RuntimeError(f"Metadata checksum mismatch for {output}")
    catalog = json.loads((REPOSITORY_ROOT / "datasets" / "catalog.json").read_text(encoding="utf-8"))
    relative_output = str(output.relative_to(REPOSITORY_ROOT))
    matching = [
        entry for entry in catalog["datasets"] if entry["data_path"] == relative_output
    ]
    if len(matching) != 1 or matching[0]["sha256"] != output_hash:
        raise RuntimeError(f"Catalog checksum is missing or stale for {relative_output}")

    labels, rebuilt, energies = _build()
    saved = load_hamiltonian_npz(str(output))
    if list(np.asarray(saved.labels, dtype=float)) != labels:
        raise RuntimeError("Saved geometry label does not match the source")
    if not np.allclose(saved.ref_energies, energies, atol=1e-12, rtol=0.0):
        raise RuntimeError("Saved CASCI reference does not match the source")
    if [len(terms) for terms in saved.hs] != variant_metadata["pauli_term_counts"]:
        raise RuntimeError("Saved Pauli term count does not match metadata")
    difference = max_pauli_coefficient_difference(saved.hs[0], rebuilt[0])
    if difference > atol:
        raise RuntimeError(f"Maximum Pauli coefficient difference is {difference:.3e}")
    print(
        f"PASS; max coefficient difference={difference:.3e}; sha256={output_hash}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify the checked-in archive")
    mode.add_argument("--write", action="store_true", help="Write the archive")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.write and (args.output_root is not None or args.force):
        raise ValueError("--output-root and --force require --write")
    if args.write:
        output_root = HERE if args.output_root is None else args.output_root.resolve()
        generate(output_root=output_root, force=args.force)
    else:
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
