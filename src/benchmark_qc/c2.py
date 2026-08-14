"""Convenience accessors for checked-out C2 benchmark data."""

from __future__ import annotations

from pathlib import Path

from .hamiltonian_test import load_hamiltonian_npz
from .integral_dataset import load_spatial_integral_archive
from .paths import BENCHMARKQC_ROOT


DATASET_ROOT = BENCHMARKQC_ROOT / "C2" / "cas8e8o_augccpvtz"
HAMILTONIAN_PATH = DATASET_ROOT / "casci_natural_orbitals" / "C2_PES_H.npz"
SOURCE_PATH = (
    DATASET_ROOT
    / "casci_natural_orbitals"
    / "inputs"
    / "stretched_r2p2000.npz"
)


def load_hamiltonian():
    """Load the trusted legacy PennyLane-object C2 archive."""

    return load_hamiltonian_npz(str(HAMILTONIAN_PATH))


def load_source_integrals():
    """Load the portable numeric C2 integral/orbital archive."""

    return load_spatial_integral_archive(SOURCE_PATH)
