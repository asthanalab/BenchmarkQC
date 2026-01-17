"""Beginner-friendly sanity check for the saved U2 Hamiltonian.

This folder contains a saved PES file with:
- `labels`: scanned geometry labels (e.g., bond length)
- `Hs`: the corresponding *qubit Hamiltonians* (PennyLane operators)
- `casci_energies`: reference energies from the same active-space CASCI/FCI

This script verifies (for one chosen point) that:
- If we diagonalize the saved qubit Hamiltonian, its ground-state energy matches
  the stored reference energy (within a tolerance).

Run examples:
  python U2/test_u2_hamiltonian.py --index 0
  python U2/test_u2_hamiltonian.py --bond 2.48

Notes:
- For U2 here the Hamiltonian is 12 qubits (dimension 4096), so we use a sparse
  eigensolver by default to avoid building a 4096x4096 dense matrix.
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

    import pennylane as qml

    n_qubits = _infer_n_qubits(terms)
    dim = 2**n_qubits
    wire_order = list(range(n_qubits))

    use_sparse = dim > 1024

    if not use_sparse:
        h_mat = np.zeros((dim, dim), dtype=np.complex128)
        for term in terms:
            h_mat += qml.matrix(term, wire_order=wire_order)
        h_mat = (h_mat + h_mat.conj().T) / 2
        e0 = float(np.min(np.linalg.eigvalsh(h_mat)).real)
        return e0, f"dense (dim={dim})"

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    h_sparse = csr_matrix((dim, dim), dtype=np.complex128)
    for term in terms:
        if hasattr(term, "sparse_matrix"):
            h_sparse = h_sparse + term.sparse_matrix(wire_order=wire_order)
        else:
            h_sparse = h_sparse + csr_matrix(qml.matrix(term, wire_order=wire_order))

    h_sparse = (h_sparse + h_sparse.getH()) * 0.5

    e0 = float(eigsh(h_sparse, k=1, which="SA", return_eigenvectors=False)[0].real)
    return e0, f"sparse-eigsh (dim={dim})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="U2/U2_PES_H.npz",
        help="Path to the saved PES npz (default: U2/U2_PES_H.npz)",
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

    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    labels = data["labels"]
    hs = data["Hs"]
    casci_energies = data["casci_energies"]

    i = _pick_point_index(labels, index=args.index, bond=args.bond)
    chosen_label = float(labels[i])

    terms = hs[i]
    e0, method = _ground_energy_from_terms(terms)

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
