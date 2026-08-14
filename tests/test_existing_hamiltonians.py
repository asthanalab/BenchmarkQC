from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_qc.hamiltonian_test import ground_energy_from_terms, load_hamiltonian_npz


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "nelec", "spin"),
    (
        ("datasets/benchmarkQC/N2/N2_PES_H1.npz", 4, 0),
        ("datasets/benchmarkQC/FeS/FeS_PES_H.npz", 6, 4),
        ("datasets/benchmarkQC/U2/U2_PES_H.npz", 6, 0),
    ),
)
def test_existing_first_point_matches_reference(
    relative_path: str,
    nelec: int,
    spin: int,
) -> None:
    data = load_hamiltonian_npz(str(ROOT / relative_path))
    energy, _ = ground_energy_from_terms(data.hs[0], nelec=nelec, spin=spin)
    assert energy == pytest.approx(float(data.ref_energies[0]), abs=1e-6)
