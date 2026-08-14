from __future__ import annotations

import numpy as np
import pytest

from benchmark_qc.integral_dataset import (
    EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE,
    jordan_wigner_terms_from_integrals,
    load_spatial_integral_archive,
)


def _write_archive(
    path,
    *,
    active_indices=None,
    reference=None,
    two_body=None,
    extra=None,
) -> None:
    payload = {
        "geometry_angstrom": np.asarray(
            [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]
        ),
        "one_body_integrals": np.eye(2),
        "two_body_integrals": (
            np.zeros((2, 2, 2, 2)) if two_body is None else two_body
        ),
        "core_constant": np.asarray([0.0]),
        "active_mo_indices": np.asarray(
            [0, 1] if active_indices is None else active_indices
        ),
        "reference_determinant": np.asarray(
            [True, True, False, False] if reference is None else reference
        ),
    }
    payload.update(extra or {})
    np.savez_compressed(path, **payload)


def test_integral_archive_rejects_noninteger_active_indices(tmp_path) -> None:
    path = tmp_path / "bad_indices.npz"
    _write_archive(path, active_indices=np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="integer dtype"):
        load_spatial_integral_archive(path)


def test_integral_archive_rejects_nonboolean_reference_values(tmp_path) -> None:
    path = tmp_path / "bad_reference.npz"
    _write_archive(path, reference=np.asarray([1, 2, 0, 0]))
    with pytest.raises(ValueError, match="Boolean/0/1"):
        load_spatial_integral_archive(path)


def test_integral_archive_rejects_broken_chemist_symmetry(tmp_path) -> None:
    path = tmp_path / "bad_eri.npz"
    two_body = np.zeros((2, 2, 2, 2))
    two_body[0, 1, 0, 0] = 1.0
    _write_archive(path, two_body=two_body)
    with pytest.raises(ValueError, match="chemist-ERI symmetry"):
        load_spatial_integral_archive(path)


def test_explicit_projector_coefficients_regenerate_hamiltonian(tmp_path) -> None:
    path = tmp_path / "explicit_projector.npz"
    _write_archive(
        path,
        active_indices=np.empty(0, dtype=np.int64),
        extra={
            "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
            "selected_mp2_natural_coeff": np.empty((2, 0)),
            "pre_hf_active_mo_coeff": np.eye(2),
            "pre_hf_active_coefficient_source": np.asarray(
                EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE
            ),
            "pre_hf_projected_mp2_natural_occupation_data": np.diag([1.0, 1.0]),
        },
    )

    archive = load_spatial_integral_archive(path)
    assert archive.active_mo_indices.size == 0
    assert archive.pre_hf_active_coefficient_source == (
        EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE
    )
    terms = jordan_wigner_terms_from_integrals(archive, cutoff=1e-14)
    assert terms.ndim == 1
    assert terms.size > 0


def test_explicit_projector_coefficients_require_source_provenance(tmp_path) -> None:
    path = tmp_path / "missing_projector_source.npz"
    _write_archive(
        path,
        active_indices=np.empty(0, dtype=np.int64),
        extra={
            "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
            "selected_mp2_natural_coeff": np.empty((2, 0)),
            "pre_hf_active_mo_coeff": np.eye(2),
            "pre_hf_projected_mp2_natural_occupation_data": np.diag([1.0, 1.0]),
        },
    )
    with pytest.raises(ValueError, match="coefficient_source provenance"):
        load_spatial_integral_archive(path)


def test_projector_source_rejects_fabricated_natural_indices(tmp_path) -> None:
    path = tmp_path / "fabricated_indices.npz"
    _write_archive(
        path,
        active_indices=np.asarray([0, 1], dtype=np.int64),
        extra={
            "selected_mp2_natural_indices": np.asarray([0, 1], dtype=np.int64),
            "selected_mp2_natural_coeff": np.eye(2),
            "pre_hf_active_mo_coeff": np.eye(2),
            "pre_hf_active_coefficient_source": np.asarray(
                EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE
            ),
            "pre_hf_projected_mp2_natural_occupation_data": np.diag([1.0, 1.0]),
        },
    )
    with pytest.raises(ValueError, match="fabricated MP2-natural orbital indices"):
        load_spatial_integral_archive(path)


def test_explicit_projector_rejects_nonfinite_projected_occupations(tmp_path) -> None:
    path = tmp_path / "nonfinite_projected_occupations.npz"
    _write_archive(
        path,
        active_indices=np.empty(0, dtype=np.int64),
        extra={
            "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
            "selected_mp2_natural_coeff": np.empty((2, 0)),
            "pre_hf_active_mo_coeff": np.eye(2),
            "pre_hf_active_coefficient_source": np.asarray(
                EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE
            ),
            "pre_hf_projected_mp2_natural_occupation_data": np.diag([1.0, np.nan]),
        },
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_spatial_integral_archive(path)


def test_explicit_projector_rejects_fabricated_selected_coefficients(tmp_path) -> None:
    path = tmp_path / "fabricated_selected_coefficients.npz"
    _write_archive(
        path,
        active_indices=np.empty(0, dtype=np.int64),
        extra={
            "selected_mp2_natural_indices": np.empty(0, dtype=np.int64),
            "selected_mp2_natural_coeff": np.eye(2),
            "pre_hf_active_mo_coeff": np.eye(2),
            "pre_hf_active_coefficient_source": np.asarray(
                EXPLICIT_PROJECTOR_COEFFICIENT_SOURCE
            ),
            "pre_hf_projected_mp2_natural_occupation_data": np.diag([1.0, 1.0]),
        },
    )
    with pytest.raises(ValueError, match="empty selected_mp2_natural_coeff"):
        load_spatial_integral_archive(path)
