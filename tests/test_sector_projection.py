from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from benchmark_qc.hamiltonian_test import sector_basis_indices, sector_matrix_from_terms


def _qml_dense_sector(terms: np.ndarray, *, n_qubits: int, nelec: int, spin: int) -> np.ndarray:
    """Slice PennyLane's big-endian dense matrix using the utility's wire-bit states."""

    operator = qml.sum(*terms)
    full = np.asarray(qml.matrix(operator, wire_order=range(n_qubits)), dtype=complex)
    wire_bit_states = sector_basis_indices(n_qubits=n_qubits, nelec=nelec, spin=spin)
    dense_indices = np.asarray(
        [
            sum(((int(state) >> wire) & 1) << (n_qubits - 1 - wire) for wire in range(n_qubits))
            for state in wire_bit_states
        ],
        dtype=int,
    )
    return full[np.ix_(dense_indices, dense_indices)]


@pytest.mark.parametrize(("nelec", "spin"), ((2, 0), (1, 1), (1, -1)))
def test_direct_sector_projection_matches_full_pennylane_matrix(nelec: int, spin: int) -> None:
    terms = np.asarray(
        [
            qml.s_prod(0.7, qml.Identity(0)),
            qml.s_prod(0.2, qml.PauliZ(0)),
            qml.s_prod(-0.3, qml.PauliZ(1)),
            qml.s_prod(0.4, qml.PauliX(0) @ qml.PauliX(2)),
            qml.s_prod(0.4, qml.PauliY(0) @ qml.PauliY(2)),
            qml.s_prod(-0.15, qml.PauliX(1) @ qml.PauliX(3)),
            qml.s_prod(-0.15, qml.PauliY(1) @ qml.PauliY(3)),
        ],
        dtype=object,
    )
    projected = sector_matrix_from_terms(terms, nelec=nelec, spin=spin).toarray()
    expected = _qml_dense_sector(terms, n_qubits=4, nelec=nelec, spin=spin)
    assert np.max(np.abs(projected - expected)) <= 1e-14


def test_direct_sector_projection_rejects_nonhermitian_coefficients() -> None:
    terms = np.asarray([qml.s_prod(1j, qml.PauliZ(0))], dtype=object)
    with pytest.raises(ValueError, match="not Hermitian"):
        sector_matrix_from_terms(terms, nelec=1, spin=1)
