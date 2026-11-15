import json
from pathlib import Path
import numpy as np
import pytest
from benchmark_qc.n2_cas6e6o import GRID, case_id, validate_system

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('distance',GRID)
def test_full_input_and_reference(distance):
    root = ROOT/'datasets/benchmarkQC/systems'/case_id(distance)
    with np.load(root/'inputs/source_integrals.npz',allow_pickle=False) as raw:
        assert raw['full_mo_coeff'].shape == (28,28)
        assert raw['active_mo_coeff'].shape == (28,6)
        assert raw['reference_uhf_density'].shape == (2,28,28)
        assert np.linalg.norm(raw['geometry_angstrom'][1]-raw['geometry_angstrom'][0]) == pytest.approx(distance,abs=1e-12)
        assert int(raw['ncore']) == 4
        assert int(raw['active_electrons']) == 6
        assert int(raw['active_spatial_orbitals']) == 6
    result = validate_system(root)
    assert result['reference_error_hartree'] < 1e-9
    meta = json.loads((root/'metadata.json').read_text())
    assert meta['provenance']['original_figure1_hamiltonian_identity_established'] is False
    with np.load(root/'inputs/source_integrals.npz',allow_pickle=False) as raw:
        assert np.allclose(raw['geometry_angstrom'],meta['geometry']['coordinates_angstrom'],atol=1e-12)
    assert meta['si_validation']['table_case'] == 18


def test_requested_geometry_grid_is_complete():
    assert GRID == (0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.5,2.8,3.0)


def test_cisd_references_are_certified_singlets():
    check = json.loads((ROOT/'datasets/benchmarkQC/source/molecules/N2/cas6e6o_ccpvdz/cisd_validation.json').read_text())
    assert check['status'] == 'passed' and len(check['points']) == len(GRID)
    for record in check['points']:
        assert abs(record['spin_square']) < 1e-10
        assert record['residual_norm'] < 1e-10
        case = 'n2_ccpvdz_cas6e6o_fixed_size_asf_'+record['geometry_id']
        meta = json.loads((ROOT/'datasets/benchmarkQC/systems'/case/'metadata.json').read_text())
        assert meta['reference_results']['cisd_energy_hartree'] == record['energy']
