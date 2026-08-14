"""Shared utilities to test saved PennyLane qubit Hamiltonians.

The saved `.npz` files in this repo store Hamiltonians as object arrays of PennyLane
Pauli terms. This module provides:
- loading helpers
- converting/diagonalizing to get the ground energy
- optional restriction to a fixed (N_alpha, N_beta) sector for open-shell systems

Assumption used across these benchmarks:
- even qubit indices are alpha spin-orbitals
- odd  qubit indices are beta  spin-orbitals

With PySCF convention: `spin = N_alpha - N_beta` (i.e., `spin = 2S`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NPZData:
    labels: np.ndarray
    hs: np.ndarray
    ref_energies: np.ndarray


def load_hamiltonian_npz(npz_path: str) -> NPZData:
    """Load `labels`, `Hs`, `casci_energies` from a saved PES `.npz`."""

    with np.load(npz_path, allow_pickle=True) as data:
        missing = {"labels", "Hs", "casci_energies"}.difference(data.files)
        if missing:
            raise ValueError(f"Missing required NPZ arrays: {sorted(missing)}")
        return NPZData(
            labels=data["labels"].copy(),
            hs=data["Hs"].copy(),
            ref_energies=data["casci_energies"].copy(),
        )


def pick_point_index(labels: np.ndarray, *, index: int | None, bond: float | None) -> int:
    """Pick point index by explicit `index` or nearest `bond` value."""

    if (index is None) == (bond is None):
        raise ValueError("Specify exactly one of --index or --bond")

    if index is not None:
        if index < 0 or index >= len(labels):
            raise IndexError(f"index {index} out of range (0..{len(labels)-1})")
        return int(index)

    assert bond is not None
    labels_f = np.array(labels, dtype=float)
    return int(np.argmin(np.abs(labels_f - float(bond))))


def infer_n_qubits(terms: np.ndarray) -> int:
    """Infer number of qubits from PennyLane term wire indices."""

    max_wire = -1
    for term in terms:
        wires = getattr(term, "wires", None)
        if wires is None:
            continue
        try:
            wire_list = list(wires)
        except Exception:
            continue
        if wire_list:
            max_wire = max(max_wire, max(wire_list))

    if max_wire < 0:
        raise ValueError("Could not infer number of qubits from Hamiltonian terms.")

    return max_wire + 1


def sector_basis_indices(*, n_qubits: int, nelec: int, spin: int) -> np.ndarray:
    """Return computational-basis indices matching fixed (N_alpha, N_beta).

    PySCF convention: `spin = N_alpha - N_beta`.
    """

    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Invalid (nelec={nelec}, spin={spin}): nelec+spin must be even to form integer N_alpha/N_beta"
        )

    n_alpha = (nelec + spin) // 2
    n_beta = (nelec - spin) // 2
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(f"Invalid (nelec={nelec}, spin={spin}): implies negative N_alpha/N_beta")

    alpha_positions = [i for i in range(n_qubits) if i % 2 == 0]
    beta_positions = [i for i in range(n_qubits) if i % 2 == 1]
    if n_alpha > len(alpha_positions) or n_beta > len(beta_positions):
        raise ValueError(
            f"Invalid sector: requested (N_alpha={n_alpha}, N_beta={n_beta}) but have only "
            f"{len(alpha_positions)} alpha and {len(beta_positions)} beta spin-orbitals"
        )

    dim = 2**n_qubits
    keep: list[int] = []
    for state_index in range(dim):
        a = 0
        b = 0
        for q in alpha_positions:
            a += (state_index >> q) & 1
        for q in beta_positions:
            b += (state_index >> q) & 1
        if a == n_alpha and b == n_beta:
            keep.append(state_index)

    if not keep:
        raise ValueError("No basis states found for the requested (nelec, spin) sector.")

    return np.array(keep, dtype=int)


def sector_matrix_from_terms(
    terms: np.ndarray,
    *,
    nelec: int,
    spin: int,
    hermiticity_atol: float = 1e-10,
):
    """Project PennyLane Pauli terms directly into a fixed electron/spin sector.

    This avoids constructing the full ``2**n_qubits`` matrix before slicing it,
    which is important for the 16-qubit N2 datasets.  Wire ``q`` is represented
    by bit ``q``; even wires are alpha and odd wires are beta.
    """

    import pennylane as qml
    from scipy import sparse

    n_qubits = infer_n_qubits(terms)
    basis = np.asarray(
        sector_basis_indices(n_qubits=n_qubits, nelec=nelec, spin=spin),
        dtype=np.int64,
    )
    dimension = int(len(basis))
    state_lookup = np.full(2**n_qubits, -1, dtype=np.int64)
    state_lookup[basis] = np.arange(dimension, dtype=np.int64)
    columns = np.arange(dimension, dtype=np.int64)

    coefficients: dict[object, complex] = {}
    for term in terms:
        for word, coefficient in qml.pauli.pauli_sentence(term).items():
            coefficients[word] = coefficients.get(word, 0.0j) + complex(coefficient)

    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    for word, coefficient in coefficients.items():
        if abs(coefficient) <= 1e-15:
            continue
        targets = basis.copy()
        phases = np.ones(dimension, dtype=np.complex128)
        for wire, pauli in word.items():
            wire_index = int(wire)
            occupied = ((basis >> wire_index) & 1).astype(np.int8)
            if pauli == "X":
                targets ^= 1 << wire_index
            elif pauli == "Y":
                phases *= 1j * (1 - 2 * occupied)
                targets ^= 1 << wire_index
            elif pauli == "Z":
                phases *= 1 - 2 * occupied
            else:  # pragma: no cover - PennyLane PauliWord has only X/Y/Z entries
                raise ValueError(f"Unsupported Pauli operator {pauli!r}")

        rows = state_lookup[targets]
        keep = rows >= 0
        if np.any(keep):
            row_blocks.append(rows[keep])
            column_blocks.append(columns[keep])
            data_blocks.append(coefficient * phases[keep])

    if not data_blocks:
        raise ValueError("Hamiltonian has no support in the requested sector")
    matrix = sparse.coo_matrix(
        (
            np.concatenate(data_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    antihermitian = matrix - matrix.getH()
    residual = (
        0.0
        if antihermitian.nnz == 0
        else float(np.max(np.abs(antihermitian.data)))
    )
    if residual > hermiticity_atol:
        raise ValueError(
            f"Projected Hamiltonian is not Hermitian: residual={residual:.3e}, "
            f"tolerance={hermiticity_atol:.3e}"
        )
    return (0.5 * (matrix + matrix.getH())).tocsr()


def ground_energy_from_terms(
    terms: np.ndarray,
    *,
    nelec: int | None = None,
    spin: int | None = None,
) -> tuple[float, str]:
    """Compute the Hamiltonian ground-state energy from saved PennyLane terms.

    If `nelec` and `spin` are provided, restrict diagonalization to that (N_alpha, N_beta)
    sector (needed for open-shell/high-spin comparisons).

    Returns:
      (energy, method_string)
    """

    import pennylane as qml

    n_qubits = infer_n_qubits(terms)
    dim = 2**n_qubits

    if (nelec is None) != (spin is None):
        raise ValueError("Specify both nelec and spin together, or neither")

    if nelec is not None and spin is not None:
        h_sub = sector_matrix_from_terms(terms, nelec=nelec, spin=spin)
        if h_sub.shape[0] <= 512:
            e0 = float(np.linalg.eigvalsh(h_sub.toarray())[0].real)
            solver = "dense"
        else:
            from scipy.sparse.linalg import eigsh

            v0 = np.linspace(1.0, 2.0, h_sub.shape[0], dtype=float)
            v0 /= np.linalg.norm(v0)
            e0 = float(
                eigsh(
                    h_sub,
                    k=1,
                    which="SA",
                    return_eigenvectors=False,
                    tol=1e-11,
                    maxiter=max(2000, 20 * h_sub.shape[0]),
                    v0=v0,
                )[0].real
            )
            solver = "sparse-eigsh"
        return (
            e0,
            f"{solver} sector (full_dim={dim}, subspace={h_sub.shape[0]}, "
            f"nelec={nelec}, spin={spin})",
        )

    wire_order = list(range(n_qubits))
    from scipy.sparse import csr_matrix

    h_sparse = csr_matrix((dim, dim), dtype=np.complex128)
    for term in terms:
        if hasattr(term, "sparse_matrix"):
            h_sparse = h_sparse + term.sparse_matrix(wire_order=wire_order)
        else:
            h_sparse = h_sparse + csr_matrix(qml.matrix(term, wire_order=wire_order))

    h_sparse = (h_sparse + h_sparse.getH()) * 0.5

    # Full-space ground energy
    # If the space is small enough, a dense solve is fine; otherwise use eigsh.
    if dim <= 512:
        h_dense = h_sparse.toarray()
        e0 = float(np.min(np.linalg.eigvalsh(h_dense)).real)
        return e0, f"dense (dim={dim})"

    from scipy.sparse.linalg import eigsh

    e0 = float(eigsh(h_sparse, k=1, which="SA", return_eigenvectors=False)[0].real)
    return e0, f"sparse-eigsh (dim={dim})"
