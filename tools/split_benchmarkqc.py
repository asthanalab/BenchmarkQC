"""Materialize one BenchmarkQC system for every point and active-space variant.

The historical repository stores several PES points in one NPZ archive.  The
BenchmarkQC catalog treats each ``(system, active-space/orbital variant,
geometry point)`` tuple as a separate benchmark system, so this tool creates
one-point archives and per-system metadata while retaining the original
multi-point files as source archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "datasets" / "catalog.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_slug(label: float) -> str:
    return f"r{label:.4f}".replace("-", "m").replace(".", "p")


def copy_one_point_archive(source: Path, index: int, destination: Path) -> None:
    with np.load(source, allow_pickle=True) as raw:
        labels = np.asarray(raw["labels"], dtype=object)
        hamiltonians = np.asarray(raw["Hs"], dtype=object)
        energies = np.asarray(raw["casci_energies"], dtype=np.float64)

    if not (len(labels) == len(hamiltonians) == len(energies)):
        raise ValueError(f"inconsistent point arrays in {source}")

    one_hamiltonian = np.empty(1, dtype=object)
    one_hamiltonian[0] = hamiltonians[index]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        labels=np.asarray([labels[index]], dtype=object),
        Hs=one_hamiltonian,
        casci_energies=np.asarray([energies[index]], dtype=np.float64),
    )


def load_points(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source, allow_pickle=True) as raw:
        return (
            np.asarray(raw["labels"], dtype=object),
            np.asarray(raw["casci_energies"], dtype=np.float64),
        )


def metadata_variant(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metadata_path = entry.get("metadata_path")
    if not metadata_path:
        return None, None
    metadata_file = ROOT / metadata_path
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    variants = metadata.get("variants")
    if isinstance(variants, dict):
        if entry["system"] == "C2":
            variant_name = "casci_natural_orbitals"
        elif entry["id"].endswith("_canonical"):
            variant_name = "canonical"
        elif entry["id"].endswith("_casci_natural_orbitals"):
            variant_name = "casci_natural_orbitals"
        else:
            variant_name = next(iter(variants))
        return metadata, variants[variant_name]

    if entry["system"] == "Fe2S2":
        accepted = metadata.get("accepted_variants", {})
        variant_name = entry["id"].removeprefix("fe2s2_chan30e20o_")
        return metadata, accepted.get(variant_name)
    return metadata, None


def source_integral_path(
    entry: dict[str, Any],
    index: int,
    variant: dict[str, Any] | None,
) -> Path | None:
    if variant and variant.get("source_integral_archives"):
        archive_root = ROOT / entry["data_path"].split("/", 1)[0]
        metadata_path = ROOT / entry["metadata_path"]
        archive_root = metadata_path.parent
        source_record = variant["source_integral_archives"][index]
        return archive_root / source_record["path"]

    if entry["system"] == "Fe2S2":
        source = ROOT / entry["data_path"]
        candidate = source.parent / "inputs" / "source_integrals.npz"
        if candidate.is_file():
            return candidate
    return None


def geometry_record(
    entry: dict[str, Any],
    index: int,
    label: float,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "point_index_in_source_archive": index,
        "point_label": label,
    }
    if entry["system"] != "Fe2S2":
        record["bond_length_angstrom"] = label
    if metadata:
        geometries = metadata.get("geometries", [])
        if index < len(geometries):
            record.update(geometries[index])
    return record


def materialize_entry(entry: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    source_archive = ROOT / entry["data_path"]
    labels, energies = load_points(source_archive)
    metadata, variant = metadata_variant(entry)
    source_archive_sha256 = sha256_file(source_archive)
    output_entries: list[dict[str, Any]] = []

    for index, (label_value, energy) in enumerate(zip(labels, energies)):
        label = float(label_value)
        suffix = "point0" if entry["system"] == "Fe2S2" else point_slug(label)
        case_id = f"{entry['id']}_{suffix}"
        case_root = output_root / entry["system"] / "systems" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        hamiltonian_path = case_root / "hamiltonian.npz"
        copy_one_point_archive(source_archive, index, hamiltonian_path)

        integral_source = source_integral_path(entry, index, variant)
        integral_path: Path | None = None
        if integral_source is not None:
            integral_path = case_root / "inputs" / "source_integrals.npz"
            integral_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(integral_source, integral_path)

        relative_hamiltonian = hamiltonian_path.relative_to(ROOT).as_posix()
        relative_metadata = (case_root / "metadata.json").relative_to(ROOT).as_posix()
        catalog_entry: dict[str, Any] = {
            key: value
            for key, value in entry.items()
            if key not in {"data_path", "metadata_path", "sha256", "point_count"}
        }
        catalog_entry.update(
            {
                "id": case_id,
                "case_name": f"{entry['system']} {suffix}",
                "data_path": relative_hamiltonian,
                "metadata_path": relative_metadata,
                "point_count": 1,
                "point_index": index,
                "point_label": label,
                "reference_energy_hartree": float(energy),
                "source_archive_path": entry["data_path"],
                "source_archive_sha256": source_archive_sha256,
                "sha256": sha256_file(hamiltonian_path),
            }
        )
        if entry["system"] != "Fe2S2":
            catalog_entry["bond_length_angstrom"] = label
        if integral_path is not None:
            catalog_entry["integral_data_path"] = integral_path.relative_to(ROOT).as_posix()
            catalog_entry["integral_sha256"] = sha256_file(integral_path)

        system_metadata: dict[str, Any] = {
            "schema": "benchmark-qc.system-metadata.v1",
            "system_id": case_id,
            "system": entry["system"],
            "active_space": {
                "active_electrons": entry.get("active_electrons"),
                "active_spatial_orbitals": entry.get("active_spatial_orbitals"),
                "qubits": entry.get("qubits"),
                "spin_2S": entry.get("spin_2S"),
            },
            "geometry": geometry_record(entry, index, label, metadata),
            "reference_energy_hartree": float(energy),
            "hamiltonian": {
                "path": relative_hamiltonian,
                "sha256": catalog_entry["sha256"],
                "format": "labels, Hs, casci_energies; one point",
            },
            "provenance": {
                "source_archive_path": entry["data_path"],
                "source_archive_sha256": source_archive_sha256,
                "source_catalog_id": entry["id"],
            },
        }
        if integral_path is not None:
            system_metadata["integrals"] = {
                "path": catalog_entry["integral_data_path"],
                "sha256": catalog_entry["integral_sha256"],
                "format": "pickle-free normalized spatial active-space archive",
            }
        if variant is not None:
            system_metadata["active_space_variant"] = variant
        if metadata is not None:
            system_metadata["source_metadata_path"] = entry.get("metadata_path")

        (case_root / "metadata.json").write_text(
            json.dumps(system_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_entries.append(catalog_entry)

    return output_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="require the expected 36 cases")
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if catalog.get("schema") == "benchmark-qc.catalog.v2":
        raise SystemExit(
            "catalog.json is already materialized at one point per system; "
            "restore an aggregate source catalog before rerunning this migration"
        )
    source_entries = [entry for entry in catalog["datasets"] if entry["system"] != "MolVQE21"]
    molvqe_entries = [entry for entry in catalog["datasets"] if entry["system"] == "MolVQE21"]
    benchmark_entries: list[dict[str, Any]] = []
    for entry in source_entries:
        benchmark_entries.extend(materialize_entry(entry, ROOT / "datasets" / "benchmarkQC"))

    benchmark_entries.sort(key=lambda item: item["id"])
    catalog["schema"] = "benchmark-qc.catalog.v2"
    catalog["description"] = (
        "One catalog entry per molecule, active-space/orbital variant, and geometry point."
    )
    catalog["datasets"] = benchmark_entries + molvqe_entries
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"materialized {len(benchmark_entries)} BenchmarkQC systems and {len(molvqe_entries)} MolVQE21 systems")
    if args.check and len(benchmark_entries) != 36:
        raise SystemExit(f"expected 36 BenchmarkQC systems, found {len(benchmark_entries)}")


if __name__ == "__main__":
    main()
