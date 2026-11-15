"""N2 fixed-size CAS(6e,6o)/cc-pVDZ inputs on the main Figure 1 geometry grid.

These are explicit new reconstructions, not recovered original plot inputs.
Frozen orbital coefficients define the deposited Hamiltonian without rerunning
a potentially different mean-field solution or active-space selection.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.8, 3.0)
PREFIX = 'n2_ccpvdz_cas6e6o_fixed_size_asf_'


def case_id(distance: float) -> str:
    if distance not in GRID:
        raise ValueError('Distance is not in the deposited twelve-point grid.')
    return PREFIX + f'r{distance:.4f}'.replace('.', 'p')


def exact_jordan_wigner_terms(h1, h2, core, cutoff=1e-14):
    """Map chemist-ordered spatial integrals, preserving the stated cutoff.

PennyLane 0.37's qchem helper invokes a second FermiSentence simplification
with a 1e-8 default. Build the standard fermionic sum explicitly to avoid
silently dropping coefficients above the requested integral cutoff.
"""
    import pennylane as qml
    from pennylane.fermi import FermiSentence, FermiWord
    h1, h2 = np.asarray(h1), np.asarray(h2)
    sentence = FermiSentence({FermiWord({}): float(core)})
    for p,q in np.argwhere(abs(h1)>=cutoff):
        for spin in (0,1):
            sentence[FermiWord({(0,int(2*p+spin)):'+', (1,int(2*q+spin)):'-'})] = h1[p,q]
    two = np.swapaxes(h2,1,3)
    for p,q,r,s in np.argwhere(abs(two)>=cutoff):
        for alpha,beta in ((0,0),(0,1),(1,0),(1,1)):
            word = FermiWord({(0,int(2*p+alpha)):'+', (1,int(2*q+beta)):'+',
                              (2,int(2*r+beta)):'-', (3,int(2*s+alpha)):'-'})
            sentence[word] = 0.5*two[p,q,r,s]
    operator = qml.jordan_wigner(sentence)
    return np.asarray(tuple(operator.operands),dtype=object)


def rebuild_integrals(source: Path):
    """Regenerate the active Hamiltonian from the saved basis and full MO frame."""
    from pyscf import ao2mo, gto, mcscf, scf
    with np.load(source,allow_pickle=False) as raw:
        atoms = [(str(symbol),xyz) for symbol,xyz in zip(raw['symbols'],raw['geometry_angstrom'])]
        mol = gto.M(atom=atoms,basis=json.loads(str(raw['basis_definition_json'])),
                    unit='Angstrom',charge=int(raw['charge']),spin=int(raw['spin_2S']),verbose=0)
        full = np.asarray(raw['full_mo_coeff'])
        assert int(raw['ncore']) == 4
        assert int(raw['active_electrons']) == int(raw['active_spatial_orbitals']) == 6
    overlap_error = float(np.max(abs(full.T@mol.intor('int1e_ovlp')@full-np.eye(28))))
    if overlap_error >= 1e-9:
        raise ValueError('Saved molecular orbitals are not AO-overlap orthonormal.')
    cas = mcscf.CASCI(scf.RHF(mol),6,(3,3))
    h1,core = cas.get_h1eff(full)
    h2 = ao2mo.restore(1,cas.get_h2eff(full),6)
    return np.asarray(h1),np.asarray(h2),float(core)


def validate_system(system_root: Path):
    """Validate geometry, AO reconstruction, all Pauli coefficients and sector energy."""
    from .hamiltonian_test import ground_energy_from_terms, load_hamiltonian_npz
    from .integral_dataset import load_spatial_integral_archive, max_pauli_coefficient_difference
    source = system_root/'inputs/source_integrals.npz'
    archive = load_spatial_integral_archive(source)
    h1,h2,core = rebuild_integrals(source)
    integral_errors = [float(np.max(abs(h1-archive.one_body_integrals))),
                       float(np.max(abs(h2-archive.two_body_integrals))),
                       abs(core-archive.core_constant)]
    if max(integral_errors) >= 1e-9:
        raise ValueError(f'AO reconstruction mismatch: {integral_errors}')
    rebuilt = exact_jordan_wigner_terms(h1,h2,core)
    saved = load_hamiltonian_npz(str(system_root/'hamiltonian.npz'))
    coefficient_error = max_pauli_coefficient_difference(saved.hs[0],rebuilt)
    if coefficient_error >= 1e-10:
        raise ValueError(f'Pauli coefficient mismatch: {coefficient_error}')
    energy,_ = ground_energy_from_terms(saved.hs[0],nelec=6,spin=0)
    difference = abs(float(energy)-float(saved.ref_energies[0]))
    if difference >= 1e-9:
        raise ValueError(f'Reference energy mismatch: {difference}')
    return {'ao_integral_errors':integral_errors,'pauli_coefficient_error':coefficient_error,
            'sector_ground_energy_hartree':float(energy),'reference_error_hartree':difference}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    rows = []
    for distance in GRID:
        root = args.repository/'datasets/benchmarkQC/systems'/case_id(distance)
        row = {'distance_angstrom':distance,**validate_system(root)}
        rows.append(row)
        print(json.dumps(row),flush=True)
    print(json.dumps({'status':'passed','points':len(rows)}))


if __name__ == '__main__':
    main()
