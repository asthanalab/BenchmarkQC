import basis_set_exchange as bse
import pyscf
from pyscf import gto, scf, mcscf
import pennylane as qml
import numpy as np
import os
import numpy as np 
import sympy as sp
from sympy import symbols, solve
import pyscf
pyscf.__config__.B3LYP_WITH_VWN5 = False
from pathlib import Path

# The Mole class is used to define molecular information in PySCF.
from pyscf.gto import Mole
from pyscf import scf

# logger contains definitions of verbosity levels for PySCF.
from pyscf.lib import logger

# Functionality for (state-averaged) CASSCF.
from pyscf.mcscf import CASSCF, CASCI, state_average_mix
from pyscf.fci.direct_spin1 import FCISolver
from pyscf.fci.addons import fix_spin

# Wrapper functions to perform selection for variable and fixed active space sizes
from asf.wrapper import find_from_mol, find_from_scf, sized_space_from_mol, sized_space_from_scf
from asf.scf import stable_scf
# Various utility functions...
from asf.utility import compare_active_spaces, show_mos_grid, pictures_Jmol

from asf import ASFDMRG
from asf.visualization import draw_pair_information
from asf.preselection import MP2NatorbPreselection, MP2PairinfoPreselection
from asf.scf import stable_scf
from asf.utility import pictures_Jmol
import sys, os, contextlib

@contextlib.contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as f:
            sys.stdout = f
            yield
    finally:
        sys.stdout = saved_stdout


def H_gen(basis_input, elements, geom, spin, charge, ncas, nelecas,
          save=True, savefile="H_data.npz", geom_id=None):
    basis = bse.get_basis(basis_input, elements=elements, fmt="nwchem")
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    mol.verbose = 0        # global PySCF verbosity
    mol.output = None      # ensure no log file is written

    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.stdout = None # disable SCF printing
    mf.kernel()

 
    n_orb = mf.mo_coeff.shape[0]

    with suppress_stdout():
        active_space = find_from_mol(mol, max_norb=ncas, verbose = 0)

    #This CASCI is taken from pyscf
    mycas = CASCI(mol, ncas=active_space.norb, nelecas=active_space.nel)


    mo_guess = mycas.sort_mo(active_space.mo_list, active_space.mo_coeff, base=0)

    mycas.kernel(mo_coeff=mo_guess, verbose=0) 
    cas_energy = mycas.kernel()[0]

    #Getting the Hamiltonian
    one_mo, ecore = mycas.get_h1eff(mycas.mo_coeff)
    h2ecas = mycas.get_h2eff(mycas.mo_coeff)
    two_mo = pyscf.ao2mo.restore("1", h2ecas, norb=ncas)
    two_mo = np.swapaxes(two_mo, 1, 3)
    core_constant = np.array([ecore])
    H_fermionic = qml.qchem.fermionic_observable(
        core_constant, one_mo, two_mo, cutoff=1e-20
    )
    H = qml.jordan_wigner(H_fermionic)
    if save:
        # use geom_id if given, else store the raw geom string
        if geom_id is None:
            label = geom
        else:
            label = geom_id

        if os.path.exists(savefile):
            data = np.load(savefile, allow_pickle=True)
            labels = list(data["labels"])
            Hs = list(data["Hs"])
            Es = list(data["casci_energies"])
        else:
            labels, Hs, Es = [], [], []

        labels.append(label)
        Hs.append(H)
        Es.append(cas_energy)

        np.savez_compressed(
            savefile,
            labels=np.array(labels, dtype=object),
            Hs=np.array(Hs, dtype=object),
            casci_energies=np.array(Es),
        )
    #Code to generate the operator pool 



    electrons = nelecas
    orbitals = ncas * 2


    N_alpha, N_beta = sp.symbols('N_alpha N_beta')
    eq1 = N_alpha + N_beta - electrons
    eq2 = N_alpha - N_beta - mol.spin

    soln = solve((eq1, eq2), (N_alpha, N_beta), dict = True)
    N_alpha = soln[0][N_alpha]
    N_beta = soln[0][N_beta]

    alpha = []
    beta = []

    alpha_occ = []
    alpha_virt = []

    beta_occ = []
    beta_virt = []

    assert N_alpha + N_beta == electrons, "No of electrons matches with Nalpha + Nbeta"

    for i in range(orbitals):
        if i % 2 == 0:
            alpha.append(i)
        else: 
            beta.append(i)
    
    for i in range(N_alpha):
        alpha_occ.append(alpha[i])

    for i in range(N_beta):
        beta_occ.append(beta[i])

    for a in alpha:
        if a not in alpha_occ:
            alpha_virt.append(a)

    for b in beta:
        if b not in beta_occ:
            beta_virt.append(b)


    delta_sz = 0
    singles = []

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )


    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    # Singles 
    for r in (alpha_occ):
        for p in (alpha_virt):
            if sz[p] - sz[r] == delta_sz:

                singles.append([r,p])

    for r in (beta_occ):
        for p in (beta_virt):
            if sz[p] - sz[r] == delta_sz:

                singles.append([r,p])
    #Doubles
    doubles = [
        sorted([s, r, q, p])
        for s in (alpha_occ)
        for r in (beta_occ)
        for q in (alpha_virt)
        for p in (beta_virt)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    op1 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})for x in doubles]
    operator_pool = (op1) + (op2)  #Operator pool - Singles and Doubles


    return H, cas_energy, operator_pool
