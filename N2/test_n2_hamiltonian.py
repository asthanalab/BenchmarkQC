"""Beginner-friendly sanity check for the saved N2 Hamiltonian.

This repo stores, for each bond length, a *qubit Hamiltonian* (PennyLane operator)
and a reference energy. This script verifies that the stored Hamiltonian produces
the same ground-state energy when we diagonalize it exactly.

What this script does (mirrors the notebook):
1) Load the npz file (contains: labels, Hs, casci_energies)
2) Choose one point by index OR by bond length (nearest match)
3) Convert the stored Pauli-term representation into a dense matrix
4) Compute the ground-state energy by exact diagonalization
5) Compare to the stored reference energy

Run examples:
  python N2/test_n2_hamiltonian.py --index 0
  python N2/test_n2_hamiltonian.py --bond 1.4

Interpretation:
- The diagonalization energy is the *exact* ground energy of the saved qubit Hamiltonian.
- The stored `casci_energies[i]` should match within numerical tolerance.
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


def _terms_to_dense_matrix(terms: np.ndarray) -> np.ndarray:
    """Convert a saved list/array of PennyLane Pauli terms into a dense Hermitian matrix."""

    # The `Hs[i]` entry saved in the npz is an object-array of PennyLane terms
    # like: (coeff) * (Z(0) @ X(1) @ ...).
    try:
        import pennylane as qml
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This script needs PennyLane to convert the stored Hamiltonian into a matrix. "
            "Make sure `pennylane` is installed in your environment."
        ) from e

    n_qubits = _infer_n_qubits(terms)
    wire_order = list(range(n_qubits))

    # For N2 in this workspace, n_qubits is typically 8 => matrix is 256x256.
    dim = 2**n_qubits
    h_mat = np.zeros((dim, dim), dtype=np.complex128)

    # Sum the dense matrices for each term.
    for term in terms:
        h_mat += qml.matrix(term, wire_order=wire_order)

    # Symmetrize to remove tiny numerical non-Hermiticity.
    return (h_mat + h_mat.conj().T) / 2


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="N2/N2_PES_H1.npz",
        help="Path to the saved PES npz (default: N2/N2_PES_H1.npz)",
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

    # -------------------- Load the file (same as notebook) --------------------
    data = np.load(args.npz, allow_pickle=True)
    labels = data["labels"]
    hs = data["Hs"]
    casci_energies = data["casci_energies"]

    # -------------------- Choose which geometry point ------------------------
    i = _pick_point_index(labels, index=args.index, bond=args.bond)
    chosen_label = float(labels[i])

    # -------------------- Pull out the saved Hamiltonian ---------------------
    # `hs[i]` is an array of PennyLane terms (object dtype).
    terms = hs[i]

    # -------------------- Convert to matrix and diagonalize ------------------
    h_mat = _terms_to_dense_matrix(terms)
    # Hermitian eigensolver; ground state is the smallest eigenvalue.
    e0 = float(np.min(np.linalg.eigvalsh(h_mat)).real)

    # -------------------- Compare to stored reference energy -----------------
    e_ref = float(casci_energies[i])
    abs_diff = abs(e0 - e_ref)

    # -------------------- Pretty output --------------------------------------
    print(f"NPZ: {args.npz}")
    print(f"Chosen point index:         {i}")
    print(f"Point label (bond length):  {chosen_label}")
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
