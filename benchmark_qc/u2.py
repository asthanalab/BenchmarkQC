"""U2 Hamiltonian generation helpers."""

from __future__ import annotations

import os

import numpy as np
import pennylane as qml
import pyscf
from pyscf import gto, mcscf, scf
from pyscf.fci import addons, direct_spin1
from pyscf.mcscf.addons import project_init_guess


def build_u2_reference(basis_input, ncas, nelecas):
    """Build a reference U2 CASSCF calculation to get reference MOs for projection."""

    mol_ref = gto.Mole()
    mol_ref.atom = """
    U 0 0 0
    U 0 0 2.5
    """
    mol_ref.basis = basis_input
    mol_ref.charge = 0
    mol_ref.spin = 0
    mol_ref.symmetry = False
    mol_ref.build()
    mol_ref.verbose = 0

    mf_ref = scf.ROHF(mol_ref).sfx2c1e()
    mf_ref.level_shift = 0.5
    mf_ref.diis_space = 12
    mf_ref.max_cycle = 100
    mf_ref.verbose = 0
    mf_ref.kernel()
    if not mf_ref.converged:
        mf_ref = scf.newton(mf_ref).run()

    mycas_ref = mcscf.CASSCF(mf_ref, ncas, nelecas)
    mycas_ref.fix_spin_(ss=0.0)
    mycas_ref.verbose = 0
    mycas_ref.kernel()

    return mol_ref, mycas_ref.mo_coeff


def H_gen(
    basis_input,
    elements,
    geom,
    spin,
    charge,
    ncas,
    nelecas,
    *,
    mol_ref,
    mo_ref,
    save: bool = True,
    savefile: str = "H_data.npz",
    geom_id=None,
):
    """Generate a qubit Hamiltonian H and CASCI energy for U2.

    Mirrors the original `U2/U2_ham1.py` implementation.
    """

    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis_input
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.build()
    mol.verbose = 0

    mf = scf.ROHF(mol).sfx2c1e()
    mf.level_shift = 0.5
    mf.diis_space = 12
    mf.max_cycle = 100
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run()

    mycas = mcscf.CASCI(mf, ncas, nelecas)
    mo_proj = project_init_guess(mycas, mo_ref, prev_mol=mol_ref)

    fcis = direct_spin1.FCI(mol)
    fcis.spin = 0
    fcis.nroots = 1
    fcis = addons.fix_spin_(fcis, ss=0.0, shift=0.8)
    mycas.fcisolver = fcis
    mycas.fix_spin_(ss=0.0)

    h1ecas, ecore = mycas.get_h1eff(mo_proj)
    h2ecas = mycas.get_h2eff(mo_proj)
    mycas.verbose = 0
    casci_energy = mycas.kernel(mo_coeff=mo_proj)[0]

    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)

    H_fermionic = qml.qchem.fermionic_observable(np.array([ecore]), h1ecas, two_mo, cutoff=1e-20)
    H = qml.jordan_wigner(H_fermionic)

    if save:
        label = geom if geom_id is None else geom_id

        if os.path.exists(savefile):
            data = np.load(savefile, allow_pickle=True)
            labels = list(data["labels"])
            Hs = list(data["Hs"])
            Es = list(data["casci_energies"])
        else:
            labels, Hs, Es = [], [], []

        labels.append(label)
        Hs.append(H)
        Es.append(casci_energy)

        np.savez_compressed(
            savefile,
            labels=np.array(labels, dtype=object),
            Hs=np.array(Hs, dtype=object),
            casci_energies=np.array(Es),
        )

    return H, casci_energy
