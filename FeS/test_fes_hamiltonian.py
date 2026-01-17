"""Beginner-friendly sanity check for the saved FeS Hamiltonian.

This folder contains a saved PES file with:
- `labels`: the scanned geometry parameter(s) (often the bond length)
- `Hs`: the corresponding *qubit Hamiltonians* (PennyLane operators)
- `casci_energies`: reference energies from the same active-space CASCI/FCI

This script verifies (for one chosen point) that:
- If we diagonalize the saved qubit Hamiltonian, its ground-state energy matches
  the stored reference energy (within a tolerance).

Run examples:
  python FeS/test_fes_hamiltonian.py --index 0
  python FeS/test_fes_hamiltonian.py --bond 2.4

Notes:
- For FeS here the Hamiltonian is 12 qubits (dimension 4096), so we use a sparse
  eigensolver by default to avoid building a 4096x4096 dense matrix.
- FeS is an open-shell high-spin state. To compare apples-to-apples against the
    stored CASCI energy, we must diagonalize *within the same (N, M_S) sector*.
    This script does that using `--nelec` and `--spin` (PySCF convention: spin = 2S).
"""

from __future__ import annotations

import argparse

import numpy as np


def _infer_n_qubits(terms: np.ndarray) -> int:
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


def _pick_point_index(labels: np.ndarray, index: int | None, bond: float | None) -> int:
    """Choose a point index either directly or by nearest bond length."""

    if (index is None) == (bond is None):
        raise ValueError("Specify exactly one of --index or --bond")

    if index is not None:
        if index < 0 or index >= len(labels):
            raise IndexError(f"index {index} out of range (0..{len(labels)-1})")
        return index

    assert bond is not None
    labels_f = np.array(labels, dtype=float)
    return int(np.argmin(np.abs(labels_f - float(bond))))


def _ground_energy_from_terms(terms: np.ndarray) -> tuple[float, str]:
    """Compute the Hamiltonian ground-state energy from saved PennyLane terms.

    Returns:
      (energy, method_used)
    """

    # PennyLane provides matrix builders for its operators.
    import pennylane as qml

    n_qubits = _infer_n_qubits(terms)
    dim = 2**n_qubits
    wire_order = list(range(n_qubits))

    # Auto-choice:
    # - dense diagonalization is simplest but memory-heavy for dim=4096
    # - sparse eigensolver is much more comfortable at this size
    use_sparse = dim > 1024

    if not use_sparse:
        h_mat = np.zeros((dim, dim), dtype=np.complex128)
        for term in terms:
            h_mat += qml.matrix(term, wire_order=wire_order)
        h_mat = (h_mat + h_mat.conj().T) / 2
        e0 = float(np.min(np.linalg.eigvalsh(h_mat)).real)
        return e0, f"dense (dim={dim})"

    # Sparse path
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    h_sparse = csr_matrix((dim, dim), dtype=np.complex128)
    for term in terms:
        # In PennyLane 0.38, sparse matrices are created from the operator object.
        if hasattr(term, "sparse_matrix"):
            h_sparse = h_sparse + term.sparse_matrix(wire_order=wire_order)
        else:
            h_sparse = h_sparse + csr_matrix(qml.matrix(term, wire_order=wire_order))

    # Symmetrize (protects against tiny numerical non-Hermiticity)
    h_sparse = (h_sparse + h_sparse.getH()) * 0.5

    # Smallest algebraic eigenvalue = ground state
    e0 = float(eigsh(h_sparse, k=1, which="SA", return_eigenvectors=False)[0].real)
    return e0, f"sparse-eigsh (dim={dim})"


def _sector_basis_indices(n_qubits: int, nelec: int, spin: int) -> np.ndarray:
    """Return computational-basis indices matching fixed (N_alpha, N_beta).

    We assume the standard ordering used throughout this repo:
    - even qubit indices are alpha spin-orbitals
    - odd  qubit indices are beta  spin-orbitals

    With PySCF convention: mol.spin = N_alpha - N_beta.
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
        # Count occupied alpha/beta spin-orbitals in this basis state
        for q in alpha_positions:
            a += (state_index >> q) & 1
        for q in beta_positions:
            b += (state_index >> q) & 1
        if a == n_alpha and b == n_beta:
            keep.append(state_index)

    if not keep:
        raise ValueError("No basis states found for the requested (nelec, spin) sector.")

    return np.array(keep, dtype=int)


def _ground_energy_in_sector(terms: np.ndarray, nelec: int, spin: int) -> tuple[float, str]:
    """Ground-state energy restricted to a fixed (nelec, spin) sector."""

    import pennylane as qml
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    n_qubits = _infer_n_qubits(terms)
    dim = 2**n_qubits
    wire_order = list(range(n_qubits))

    h_sparse = csr_matrix((dim, dim), dtype=np.complex128)
    for term in terms:
        if hasattr(term, "sparse_matrix"):
            h_sparse = h_sparse + term.sparse_matrix(wire_order=wire_order)
        else:
            h_sparse = h_sparse + csr_matrix(qml.matrix(term, wire_order=wire_order))

    h_sparse = (h_sparse + h_sparse.getH()) * 0.5

    keep = _sector_basis_indices(n_qubits=n_qubits, nelec=nelec, spin=spin)
    h_sub = h_sparse[keep][:, keep]

    # Compute smallest eigenvalue in this sector
    e0 = float(eigsh(h_sub, k=1, which="SA", return_eigenvectors=False)[0].real)
    return e0, f"sparse-eigsh sector (dim={dim}, subspace={h_sub.shape[0]}, nelec={nelec}, spin={spin})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="FeS/FeS_PES_H.npz",
        help="Path to the saved PES npz (default: FeS/FeS_PES_H.npz)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Point index in labels/Hs arrays")
    group.add_argument(
        "--bond",
        type=float,
        help="Bond length to test (chooses nearest stored label)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for energy match (Ha)",
    )
    parser.add_argument(
        "--nelec",
        type=int,
        default=6,
        help="Active-space electrons for the comparison sector (default: 6, from FeS-in.ipynb)",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=4,
        help="PySCF spin (2S) for the comparison sector (default: 4, from FeS-in.ipynb)",
    )

    args = parser.parse_args()

    # Load exactly like the notebooks do
    data = np.load(args.npz, allow_pickle=True)
    labels = data["labels"]
    hs = data["Hs"]
    casci_energies = data["casci_energies"]

    i = _pick_point_index(labels, index=args.index, bond=args.bond)
    chosen_label = float(labels[i])

    terms = hs[i]
    # IMPORTANT for open-shell systems: compare in the same (nelec, spin) sector.
    e0, method = _ground_energy_in_sector(terms, nelec=args.nelec, spin=args.spin)

    e_ref = float(casci_energies[i])
    abs_diff = abs(e0 - e_ref)

    print(f"NPZ: {args.npz}")
    print(f"Chosen point index:         {i}")
    print(f"Point label:                {chosen_label}")
    print(f"Diagonalization method:     {method}")
    print(f"Stored CASCI energy (Ha):   {e_ref:.12f}")
    print(f"Diag ground energy (Ha):    {e0:.12f}")
    print(f"|diff| (Ha):                {abs_diff:.6e}")

    if abs_diff <= args.atol:
        print(f"PASS (|diff| <= {args.atol})")
        return 0

    print(f"FAIL (|diff| > {args.atol})")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
