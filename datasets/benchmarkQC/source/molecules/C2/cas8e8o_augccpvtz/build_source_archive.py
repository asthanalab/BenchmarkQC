#!/usr/bin/env python3
"""Build the portable C2 CASCI-natural-orbital source archive.

This importer intentionally requires the two immutable QCANT source artifacts
as command-line inputs.  No workstation-specific source path is recorded in
the resulting dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np


NATURAL_INPUT_SHA256 = (
    "b556792b31e84423daea9ba47077a54a4a38986bd3b9861475bbc7a944c930b2"
)
PARENT_INPUT_SHA256 = (
    "82d69fd0a9883ea2ed9579632e8720bc48a98f2485e55c44f38a03a28685448c"
)
EXPECTED_ACTIVE_INDICES = np.arange(2, 10, dtype=np.int64)
EXPECTED_OCCUPATIONS = np.asarray(
    [
        1.9614528141136904,
        1.9438978889458896,
        1.413140623649,
        0.9784451202319453,
        0.9762048438572233,
        0.5890745691498872,
        0.07646866487866846,
        0.061315475173672966,
    ],
    dtype=float,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"Source checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def build(natural_input: Path, parent_input: Path, output: Path, *, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
    _require_hash(natural_input, NATURAL_INPUT_SHA256)
    _require_hash(parent_input, PARENT_INPUT_SHA256)

    with np.load(natural_input, allow_pickle=False) as source:
        geometry = np.asarray(source["geometry_angstrom"], dtype=float)
        one_body = np.asarray(source["one_body_integrals"], dtype=float)
        two_body = np.asarray(source["two_body_integrals"], dtype=float)
        core_constant = float(np.asarray(source["core_constant"]).reshape(-1)[0])
        active_electrons = int(np.asarray(source["active_electrons"]).reshape(-1)[0])
        active_orbitals = int(
            np.asarray(source["active_spatial_orbitals"]).reshape(-1)[0]
        )
        geometry_id = str(np.asarray(source["geometry_id"]).reshape(-1)[0])
        reference = np.asarray(source["reference_determinant"], dtype=bool)
        rotation = np.asarray(source["natural_orbital_rotation"], dtype=float)
        occupations = np.asarray(source["casci_natural_occupations"], dtype=float)

    with np.load(parent_input, allow_pickle=False) as parent:
        parent_geometry = np.asarray(parent["geometry_angstrom"], dtype=float)
        active_indices = np.asarray(parent["active_mo_indices"], dtype=np.int64)
        selected_indices = np.asarray(
            parent["selected_mp2_natural_indices"], dtype=np.int64
        )
        selected_coeff = np.asarray(parent["selected_mp2_natural_coeff"], dtype=float)
        pre_hf_coeff = np.asarray(parent["pre_hf_active_mo_coeff"], dtype=float)
        projected_occupations = np.asarray(
            parent["pre_hf_projected_mp2_natural_occupation_data"], dtype=float
        )
        coefficient_source = str(
            np.asarray(parent["pre_hf_active_coefficient_source"]).reshape(-1)[0]
        )
        active_hf_rotation = np.asarray(
            parent["active_hf_rotation_matrix"], dtype=float
        )
        source_active_hf_coeff = np.asarray(parent["active_hf_mo_coeff"], dtype=float)

    if geometry_id != "stretched_r2p2000":
        raise ValueError(f"Unexpected geometry identifier {geometry_id!r}")
    if active_electrons != 8 or active_orbitals != 8:
        raise ValueError("Expected the validated C2 CAS(8e,8o) source")
    if not np.allclose(geometry, parent_geometry, atol=0.0, rtol=0.0):
        raise ValueError("Natural-orbital and parent geometries disagree")
    if not np.array_equal(active_indices, EXPECTED_ACTIVE_INDICES):
        raise ValueError("Expected parent active-orbital indices 2..9")
    if not np.array_equal(selected_indices, active_indices):
        raise ValueError("Selected MP2-natural indices do not match active indices")
    if not np.array_equal(reference, np.asarray([True] * 8 + [False] * 8)):
        raise ValueError("Unexpected eight-electron closed-shell reference determinant")
    if not np.allclose(occupations, EXPECTED_OCCUPATIONS, atol=1e-13, rtol=0.0):
        raise ValueError("CASCI natural occupations do not match the validated source")
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(active_orbitals))
    )
    if orthogonality_error > 1e-12:
        raise ValueError(
            f"Natural-orbital rotation is not orthogonal: {orthogonality_error:.3e}"
        )

    natural_active_coeff = source_active_hf_coeff @ rotation
    source_casci_rdm1 = rotation @ np.diag(occupations) @ rotation.T
    natural_rdm1 = rotation.T @ source_casci_rdm1 @ rotation
    reconstruction_error = float(
        np.linalg.norm(natural_rdm1 - np.diag(occupations))
    )
    if reconstruction_error > 1e-12:
        raise ValueError(
            "Natural-orbital occupation reconstruction failed: "
            f"{reconstruction_error:.3e}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(
            temporary,
            geometry_angstrom=geometry,
            one_body_integrals=one_body,
            two_body_integrals=two_body,
            core_constant=np.asarray([core_constant], dtype=np.float64),
            active_mo_indices=active_indices,
            selected_mp2_natural_indices=selected_indices,
            selected_mp2_natural_coeff=selected_coeff,
            pre_hf_active_mo_coeff=pre_hf_coeff,
            pre_hf_projected_mp2_natural_occupation_data=projected_occupations,
            pre_hf_active_coefficient_source=np.asarray(coefficient_source),
            reference_determinant=reference,
            natural_orbital_rotation=rotation,
            natural_orbital_occupations=occupations,
            natural_active_mo_coeff=natural_active_coeff,
            source_active_hf_mo_coeff=source_active_hf_coeff,
            source_active_hf_rotation_matrix=active_hf_rotation,
            source_casci_rdm1=source_casci_rdm1,
            source_casci_rdm1_in_natural_basis=natural_rdm1,
            geometry_id=np.asarray(geometry_id),
            active_electrons=np.asarray(active_electrons, dtype=np.int64),
            active_spatial_orbitals=np.asarray(active_orbitals, dtype=np.int64),
            exact_target_information_used=np.asarray(True),
            source_natural_input_sha256=np.asarray(NATURAL_INPUT_SHA256),
            source_parent_input_sha256=np.asarray(PARENT_INPUT_SHA256),
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {output}")
    print(f"SHA-256: {sha256_file(output)}")
    print(f"Rotation orthogonality error: {orthogonality_error:.3e}")
    print(f"Natural-occupation reconstruction error: {reconstruction_error:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-input", required=True, type=Path)
    parser.add_argument("--parent-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(
        args.natural_input.resolve(),
        args.parent_input.resolve(),
        args.output.resolve(),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
