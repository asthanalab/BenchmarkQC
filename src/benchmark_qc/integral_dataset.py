"""Build legacy Benchmark-QC archives from portable spatial-integral data.

The repository's historical ``Hs`` arrays contain pickled PennyLane operator
objects.  New datasets additionally retain numeric integral/orbital archives so
that the qubit Hamiltonians can be regenerated and independently inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


INDEXED_MP2_NATURAL_COEFFICIENT_SOURCES = {
    "selected MP2 natural orbitals",
    "subset of MP2 natural orbitals",
}
EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE = (
    "explicit frozen-core-orthogonal chemical-projector coefficients"
)
EXPLICIT_ACTIVE_FRAME_COEFFICIENT_SOURCES = {
    EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE,
    "stored active-space integral frame; AO coefficients unavailable",
    "CASSCF-refined selected active-space coefficients",
    "Jordan-Wigner-equivalent active-space frame; original orbital coefficients unavailable",
}


@dataclass(frozen=True)
class SpatialIntegralArchive:
    """Validated active-space data loaded from a numeric ``.npz`` archive."""

    geometry_angstrom: np.ndarray
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    core_constant: float
    active_mo_indices: np.ndarray
    reference_determinant: np.ndarray
    pre_hf_active_mo_coeff: np.ndarray | None = None
    pre_hf_active_coefficient_source: str | None = None
    pre_hf_projected_mp2_natural_occupation_data: np.ndarray | None = None

    @property
    def n_spatial_orbitals(self) -> int:
        return int(self.one_body_integrals.shape[0])

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_spatial_orbitals


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_array(name: str, value: np.ndarray, *, imag_atol: float) -> np.ndarray:
    array = np.asarray(value)
    maximum_imaginary = (
        float(np.max(np.abs(np.asarray(array, dtype=complex).imag)))
        if array.size
        else 0.0
    )
    if maximum_imaginary > imag_atol:
        raise ValueError(
            f"{name} has imaginary component {maximum_imaginary:.3e}, "
            f"larger than {imag_atol:.3e}"
        )
    return np.asarray(array.real, dtype=float)


def _coefficient_source(value: np.ndarray, *, path: Path) -> str:
    source_array = np.asarray(value)
    if source_array.size != 1 or source_array.dtype.kind not in "US":
        raise ValueError(
            f"{path}: pre_hf_active_coefficient_source must contain one string"
        )
    scalar = source_array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        source = scalar.decode("utf-8")
    else:
        source = str(scalar)
    if not source.strip():
        raise ValueError(
            f"{path}: pre_hf_active_coefficient_source must be non-empty"
        )
    return source


def _validate_projected_occupation_data(
    value: np.ndarray,
    *,
    n_orbitals: int,
    path: Path,
) -> np.ndarray:
    occupations = _real_array(
        "pre_hf_projected_mp2_natural_occupation_data",
        value,
        imag_atol=1e-10,
    )
    if occupations.shape not in {(n_orbitals,), (n_orbitals, n_orbitals)}:
        raise ValueError(
            f"{path}: pre_hf_projected_mp2_natural_occupation_data must have "
            f"shape ({n_orbitals},) or ({n_orbitals}, {n_orbitals})"
        )
    if not np.all(np.isfinite(occupations)):
        raise ValueError(
            f"{path}: projected MP2-natural occupation data contain non-finite values"
        )
    if occupations.ndim == 2:
        if not np.allclose(occupations, occupations.T, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"{path}: projected MP2-natural occupation matrix is not symmetric"
            )
        occupation_values = np.linalg.eigvalsh(occupations)
    else:
        occupation_values = occupations
    if np.min(occupation_values) < -1e-8 or np.max(occupation_values) > 2.0 + 1e-8:
        raise ValueError(
            f"{path}: projected MP2-natural occupations leave the physical [0, 2] range"
        )
    return occupations


def load_spatial_integral_archive(
    path: str | Path,
    *,
    imag_atol: float = 1e-10,
    eri_symmetry_atol: float = 1e-9,
) -> SpatialIntegralArchive:
    """Load and validate one trusted, pickle-free integral/orbital archive."""

    required = {
        "geometry_angstrom",
        "one_body_integrals",
        "two_body_integrals",
        "core_constant",
        "active_mo_indices",
        "reference_determinant",
    }
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required arrays: {sorted(missing)}")
        geometry = np.asarray(data["geometry_angstrom"], dtype=float)
        one_body = _real_array(
            "one_body_integrals", data["one_body_integrals"], imag_atol=imag_atol
        )
        two_body = _real_array(
            "two_body_integrals", data["two_body_integrals"], imag_atol=imag_atol
        )
        core_array = _real_array(
            "core_constant", data["core_constant"], imag_atol=imag_atol
        )
        active_indices_raw = np.asarray(data["active_mo_indices"])
        reference_raw = np.asarray(data["reference_determinant"])
        selected_indices_raw = (
            np.asarray(data["selected_mp2_natural_indices"])
            if "selected_mp2_natural_indices" in data.files
            else None
        )
        selected_coeff = (
            _real_array(
                "selected_mp2_natural_coeff",
                data["selected_mp2_natural_coeff"],
                imag_atol=imag_atol,
            )
            if "selected_mp2_natural_coeff" in data.files
            else None
        )
        pre_hf_coeff = (
            _real_array(
                "pre_hf_active_mo_coeff",
                data["pre_hf_active_mo_coeff"],
                imag_atol=imag_atol,
            )
            if "pre_hf_active_mo_coeff" in data.files
            else None
        )
        coefficient_source = (
            _coefficient_source(
                data["pre_hf_active_coefficient_source"], path=archive_path
            )
            if "pre_hf_active_coefficient_source" in data.files
            else None
        )
        projected_occupation_raw = (
            np.asarray(data["pre_hf_projected_mp2_natural_occupation_data"])
            if "pre_hf_projected_mp2_natural_occupation_data" in data.files
            else None
        )

    if active_indices_raw.dtype.kind not in "iu":
        raise ValueError("active_mo_indices must use an integer dtype")
    if reference_raw.dtype.kind not in "biu" or not np.all(
        (reference_raw == 0) | (reference_raw == 1)
    ):
        raise ValueError("reference_determinant must contain only Boolean/0/1 values")
    active_indices = np.asarray(active_indices_raw, dtype=int)
    reference = np.asarray(reference_raw, dtype=bool)

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError(f"one_body_integrals must be square, got {one_body.shape}")
    n_orbitals = int(one_body.shape[0])
    expected_two_body_shape = (n_orbitals,) * 4
    if two_body.shape != expected_two_body_shape:
        raise ValueError(
            f"two_body_integrals must have shape {expected_two_body_shape}, "
            f"got {two_body.shape}"
        )
    if core_array.size != 1:
        raise ValueError(f"core_constant must contain one value, got {core_array.shape}")
    indexed_space = active_indices.shape == (n_orbitals,)
    explicit_coefficient_space = active_indices.shape == (0,)
    if not indexed_space and not explicit_coefficient_space:
        raise ValueError(
            f"active_mo_indices must have shape ({n_orbitals},) for an indexed "
            f"MP2-natural space or (0,) for an explicit coefficient-defined space; "
            f"got {active_indices.shape}"
        )
    if indexed_space and (
        np.any(active_indices < 0) or len(np.unique(active_indices)) != n_orbitals
    ):
        raise ValueError("active_mo_indices must be unique and non-negative")
    if selected_indices_raw is not None:
        if selected_indices_raw.dtype.kind not in "iu" or selected_indices_raw.ndim != 1:
            raise ValueError("selected_mp2_natural_indices must be an integer vector")
        if not np.array_equal(active_indices, np.asarray(selected_indices_raw, dtype=int)):
            raise ValueError(
                "active_mo_indices and selected_mp2_natural_indices must agree"
            )
    elif explicit_coefficient_space:
        raise ValueError(
            "explicit coefficient-defined spaces require empty "
            "selected_mp2_natural_indices"
        )
    if indexed_space:
        if coefficient_source == EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE:
            raise ValueError(
                "explicit chemical-projector coefficients cannot publish fabricated "
                "MP2-natural orbital indices"
            )
        if coefficient_source is not None and (
            coefficient_source not in INDEXED_MP2_NATURAL_COEFFICIENT_SOURCES
        ):
            raise ValueError(
                f"unrecognized indexed active-coefficient source {coefficient_source!r}"
            )
        if pre_hf_coeff is not None:
            if pre_hf_coeff.ndim != 2 or pre_hf_coeff.shape[1] != n_orbitals:
                raise ValueError(
                    "pre_hf_active_mo_coeff must have n_orbitals columns"
                )
            if not np.all(np.isfinite(pre_hf_coeff)):
                raise ValueError("pre_hf_active_mo_coeff contains non-finite values")
        if selected_coeff is not None:
            if selected_coeff.ndim != 2 or selected_coeff.shape[1] != n_orbitals:
                raise ValueError(
                    "selected_mp2_natural_coeff must have n_orbitals columns"
                )
            if pre_hf_coeff is not None and not np.allclose(
                selected_coeff,
                pre_hf_coeff,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "indexed MP2-natural coefficients disagree with the pre-HF frame"
                )
    else:
        if pre_hf_coeff is None or (
            pre_hf_coeff.ndim != 2 or pre_hf_coeff.shape[1] != n_orbitals
        ):
            raise ValueError(
                "explicit coefficient-defined spaces require a non-empty "
                "pre_hf_active_mo_coeff matrix with n_orbitals columns"
            )
        if not np.all(np.isfinite(pre_hf_coeff)):
            raise ValueError("pre_hf_active_mo_coeff contains non-finite values")
        if coefficient_source not in EXPLICIT_ACTIVE_FRAME_COEFFICIENT_SOURCES:
            raise ValueError(
                "explicit coefficient-defined spaces require recognized "
                "pre_hf_active_coefficient_source provenance"
            )
        if projected_occupation_raw is None:
            raise ValueError(
                "explicit coefficient-defined spaces require projected MP2-natural "
                "occupation data"
            )
        if selected_coeff is None or selected_coeff.ndim != 2 or (
            selected_coeff.shape[0] != pre_hf_coeff.shape[0]
            or selected_coeff.shape[1] != 0
        ):
            raise ValueError(
                "explicit coefficient-defined spaces require an empty "
                "selected_mp2_natural_coeff array with the recorded AO dimension"
            )
    projected_occupation_data = (
        _validate_projected_occupation_data(
            projected_occupation_raw,
            n_orbitals=n_orbitals,
            path=archive_path,
        )
        if projected_occupation_raw is not None
        else None
    )
    if reference.shape != (2 * n_orbitals,):
        raise ValueError(
            f"reference_determinant must have shape ({2 * n_orbitals},), "
            f"got {reference.shape}"
        )
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError(f"geometry_angstrom must have shape (n_atoms, 3), got {geometry.shape}")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry_angstrom contains non-finite values")
    if not np.all(np.isfinite(one_body)) or not np.all(np.isfinite(two_body)):
        raise ValueError("integral arrays contain non-finite values")
    if not np.all(np.isfinite(core_array)):
        raise ValueError("core_constant contains a non-finite value")
    if not np.allclose(one_body, one_body.T, atol=1e-10, rtol=0.0):
        raise ValueError("one_body_integrals are not Hermitian within tolerance")
    eri_permutations = {
        "p<->q": np.swapaxes(two_body, 0, 1),
        "r<->s": np.swapaxes(two_body, 2, 3),
        "(pq)<->(rs)": np.transpose(two_body, (2, 3, 0, 1)),
    }
    for symmetry, permuted in eri_permutations.items():
        if not np.allclose(two_body, permuted, atol=eri_symmetry_atol, rtol=0.0):
            difference = float(np.max(np.abs(two_body - permuted)))
            raise ValueError(
                f"two_body_integrals violate chemist-ERI symmetry {symmetry}; "
                f"maximum difference is {difference:.3e}"
            )

    return SpatialIntegralArchive(
        geometry_angstrom=geometry,
        one_body_integrals=one_body,
        two_body_integrals=two_body,
        core_constant=float(core_array.ravel()[0]),
        active_mo_indices=active_indices,
        reference_determinant=reference,
        pre_hf_active_mo_coeff=pre_hf_coeff,
        pre_hf_active_coefficient_source=coefficient_source,
        pre_hf_projected_mp2_natural_occupation_data=projected_occupation_data,
    )


def jordan_wigner_terms_from_integrals(
    archive: SpatialIntegralArchive,
    *,
    cutoff: float = 1e-14,
) -> np.ndarray:
    """Return PennyLane Jordan-Wigner terms for chemist-ordered spatial ERIs.

    The source two-electron tensor uses PySCF's chemist ordering.  PennyLane's
    ``fermionic_observable`` expects the index convention obtained here by
    swapping axes 1 and 3.  Spin orbitals are interleaved: even wires are alpha
    and odd wires are beta.
    """

    if cutoff < 0.0:
        raise ValueError("cutoff must be non-negative")

    import pennylane as qml

    two_body_pennylane = np.swapaxes(archive.two_body_integrals, 1, 3)
    fermionic = qml.qchem.fermionic_observable(
        np.asarray([archive.core_constant], dtype=float),
        archive.one_body_integrals,
        two_body_pennylane,
        cutoff=float(cutoff),
    )
    qubit_hamiltonian = qml.jordan_wigner(fermionic)
    operands = getattr(qubit_hamiltonian, "operands", None)
    if operands is None:
        operands = (qubit_hamiltonian,)
    return np.asarray(tuple(operands), dtype=object)


def pauli_coefficient_map(terms: Iterable[object]) -> dict[tuple[tuple[int, str], ...], complex]:
    """Convert PennyLane terms to a canonical Pauli-word coefficient map."""

    import pennylane as qml

    result: dict[tuple[tuple[int, str], ...], complex] = {}
    for term in terms:
        sentence = qml.pauli.pauli_sentence(term)
        for word, coefficient in sentence.items():
            key = tuple(sorted((int(wire), str(pauli)) for wire, pauli in word.items()))
            result[key] = result.get(key, 0.0j) + complex(coefficient)
    return {key: value for key, value in result.items() if abs(value) > 1e-15}


def max_pauli_coefficient_difference(
    left: Iterable[object],
    right: Iterable[object],
) -> float:
    """Return the largest coefficient difference between two Pauli sums."""

    left_map = pauli_coefficient_map(left)
    right_map = pauli_coefficient_map(right)
    keys = set(left_map).union(right_map)
    return max(
        (abs(left_map.get(key, 0.0j) - right_map.get(key, 0.0j)) for key in keys),
        default=0.0,
    )


def write_legacy_hamiltonian_npz(
    output_path: str | Path,
    *,
    labels: Sequence[float],
    hamiltonian_terms: Sequence[np.ndarray],
    casci_energies: Sequence[float],
) -> None:
    """Atomically write the repository's three-array legacy NPZ contract."""

    output = Path(output_path)
    if not (len(labels) == len(hamiltonian_terms) == len(casci_energies)):
        raise ValueError("labels, hamiltonian_terms, and casci_energies must align")
    if len(labels) == 0:
        raise ValueError("at least one geometry is required")

    numeric_labels = np.asarray([float(value) for value in labels], dtype=float)
    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError("labels contain non-finite values")
    if np.any(np.diff(numeric_labels) <= 0.0):
        raise ValueError("labels must be unique and strictly increasing")
    labels_array = np.asarray(numeric_labels.tolist(), dtype=object)
    energies_array = np.asarray(casci_energies, dtype=np.float64)
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("casci_energies contain non-finite values")
    hs_array = np.empty(len(hamiltonian_terms), dtype=object)
    for index, terms in enumerate(hamiltonian_terms):
        term_array = np.asarray(terms, dtype=object)
        if term_array.ndim != 1 or len(term_array) == 0:
            raise ValueError(f"Hamiltonian {index} must be a non-empty 1-D term array")
        hs_array[index] = term_array

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(
            temporary,
            labels=labels_array,
            Hs=hs_array,
            casci_energies=energies_array,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
