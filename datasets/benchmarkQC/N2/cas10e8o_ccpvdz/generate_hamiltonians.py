#!/usr/bin/env python3
"""Generate the two N2 CAS(10e,8o)/cc-pVDZ Benchmark-QC archives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Variant:
    directory: str
    description: str
    energies_hartree: tuple[float, float]


GEOMETRIES = (
    ("equilibrium_r1p0977", 1.0977),
    ("stretched_r2p0000", 2.0000),
)
VARIANTS = {
    "canonical": Variant(
        directory="canonical",
        description="canonical RHF orbitals",
        energies_hartree=(-109.03438038143877, -108.75697642204275),
    ),
    "casci_natural_orbitals": Variant(
        directory="casci_natural_orbitals",
        description="geometry-specific spin-summed exact-CASCI natural orbitals",
        energies_hartree=(-109.03438038143884, -108.75697642204210),
    ),
}


def _build_variant(name: str) -> tuple[list[float], list[np.ndarray], list[float]]:
    recipe = VARIANTS[name]
    labels: list[float] = []
    hamiltonians: list[np.ndarray] = []
    for geometry_id, bond_length in GEOMETRIES:
        source_path = HERE / recipe.directory / "inputs" / f"{geometry_id}.npz"
        archive = load_spatial_integral_archive(source_path)
        if archive.n_spatial_orbitals != 8:
            raise ValueError(f"{source_path}: expected eight active spatial orbitals")
        if not np.array_equal(archive.active_mo_indices, np.arange(2, 10)):
            raise ValueError(f"{source_path}: expected active canonical MO indices 2..9")
        expected_reference = np.asarray([True] * 10 + [False] * 6, dtype=bool)
        if not np.array_equal(archive.reference_determinant, expected_reference):
            raise ValueError(f"{source_path}: unexpected 10-electron reference determinant")
        measured_bond = float(np.linalg.norm(archive.geometry_angstrom[1] - archive.geometry_angstrom[0]))
        if abs(measured_bond - bond_length) > 1e-10:
            raise ValueError(
                f"{source_path}: geometry gives R={measured_bond}, expected {bond_length} Angstrom"
            )
        labels.append(bond_length)
        hamiltonians.append(jordan_wigner_terms_from_integrals(archive, cutoff=1e-14))
    return labels, hamiltonians, list(recipe.energies_hartree)


def generate(name: str, *, output_root: Path, force: bool) -> None:
    """Generate one variant and print its integrity summary."""

    recipe = VARIANTS[name]
    labels, hamiltonians, energies = _build_variant(name)
    output = output_root / recipe.directory / "N2_PES_H.npz"
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
        f"{name}: wrote {output}; "
        f"terms={[len(terms) for terms in hamiltonians]}; sha256={sha256_file(output)}"
    )


def _check_recorded_hashes(name: str) -> dict:
    """Verify source/output hashes in both publication metadata and catalog."""

    metadata = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    variant_metadata = metadata["variants"][name]
    for source in variant_metadata["source_integral_archives"]:
        source_path = HERE / source["path"]
        actual = sha256_file(source_path)
        if actual != source["sha256"]:
            raise RuntimeError(
                f"{name}: source checksum mismatch for {source_path}: "
                f"expected {source['sha256']}, got {actual}"
            )

    output = HERE / variant_metadata["output"]["path"]
    output_hash = sha256_file(output)
    if output_hash != variant_metadata["output"]["sha256"]:
        raise RuntimeError(
            f"{name}: metadata checksum mismatch for {output}: "
            f"expected {variant_metadata['output']['sha256']}, got {output_hash}"
        )

    catalog = json.loads((REPOSITORY_ROOT / "datasets" / "catalog.json").read_text(encoding="utf-8"))
    relative_output = str(output.relative_to(REPOSITORY_ROOT))
    matching_entries = [
        entry for entry in catalog["datasets"] if entry["data_path"] == relative_output
    ]
    if len(matching_entries) != 1:
        raise RuntimeError(f"{name}: expected one catalog entry for {relative_output}")
    if matching_entries[0]["sha256"] != output_hash:
        raise RuntimeError(f"{name}: datasets/catalog.json checksum is stale for {relative_output}")
    return variant_metadata


def check(name: str, *, atol: float = 1e-12) -> None:
    """Verify an existing archive against a fresh source-integral conversion."""

    recipe = VARIANTS[name]
    variant_metadata = _check_recorded_hashes(name)
    labels, rebuilt, energies = _build_variant(name)
    output = HERE / recipe.directory / "N2_PES_H.npz"
    saved = load_hamiltonian_npz(str(output))
    if list(np.asarray(saved.labels, dtype=float)) != labels:
        raise RuntimeError(f"{name}: saved geometry labels do not match the sources")
    if not np.allclose(saved.ref_energies, energies, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{name}: saved CASCI references do not match the recipe")
    term_counts = [len(terms) for terms in saved.hs]
    if term_counts != variant_metadata["pauli_term_counts"]:
        raise RuntimeError(f"{name}: saved term counts do not match metadata")
    differences = [
        max_pauli_coefficient_difference(saved.hs[index], rebuilt[index])
        for index in range(len(labels))
    ]
    if max(differences) > atol:
        raise RuntimeError(f"{name}: maximum Pauli coefficient difference is {max(differences):.3e}")
    print(
        f"{name}: PASS; max coefficient difference={max(differences):.3e}; "
        f"sha256={sha256_file(output)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("all",) + tuple(VARIANTS),
        default="all",
        help="Dataset variant to process (default: all)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing output without modifying it (default)",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write regenerated archives; existing files are protected by default",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination root for --write (recommended for reproducibility checks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --write to overwrite an existing output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.write and (args.output_root is not None or args.force):
        raise ValueError("--output-root and --force require --write")
    names = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    for name in names:
        if args.write:
            output_root = HERE if args.output_root is None else args.output_root.resolve()
            generate(name, output_root=output_root, force=args.force)
        else:
            check(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
