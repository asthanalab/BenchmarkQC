"""FeS Hamiltonian generation helpers."""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import pennylane as qml
import pyscf
from pyscf import ao2mo, gto, scf
from pyscf.mcscf import CASCI

pyscf.__config__.B3LYP_WITH_VWN5 = False


@contextlib.contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as f:
            sys.stdout = f
            yield
    finally:
        sys.stdout = saved_stdout


def H_gen(
    basis_input,
    elements,
    geom,
    spin,
    charge,
    ncas,
    nelecas,
    *,
    save: bool = True,
    savefile: str = "H_data.npz",
    geom_id=None,
):
    """Generate a qubit Hamiltonian H, CASCI energy, and an operator pool for FeS.

    This mirrors the original `FeS/FeS_pyscf.py` implementation.
    """

    try:
        import basis_set_exchange as bse
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `basis_set_exchange`. Install it (e.g., `pip install basis_set_exchange`)."
        ) from e

    try:
        from asf.wrapper import find_from_mol
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `asf` (active space finder). Install it to generate FeS Hamiltonians."
        ) from e

    basis = bse.get_basis(basis_input, elements=elements, fmt="nwchem")

    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    mol.verbose = 0
    mol.output = None

    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.verbose = 0
    mf.stdout = None
    mf.kernel()

    with suppress_stdout():
        active_space = find_from_mol(mol, max_norb=ncas, min_norb=3, verbose=0)

    mycas = CASCI(mol, ncas=active_space.norb, nelecas=active_space.nel)
    mo_guess = mycas.sort_mo(active_space.mo_list, active_space.mo_coeff, base=0)

    mycas.kernel(mo_coeff=mo_guess, verbose=0)
    cas_energy = mycas.kernel()[0]

    one_mo, ecore = mycas.get_h1eff(mycas.mo_coeff)
    h2ecas = mycas.get_h2eff(mycas.mo_coeff)

    # IMPORTANT: use the active-space size actually used by CASCI.
    two_mo = ao2mo.restore("1", h2ecas, norb=active_space.norb)
    two_mo = np.swapaxes(two_mo, 1, 3)

    H_fermionic = qml.qchem.fermionic_observable(np.array([ecore]), one_mo, two_mo, cutoff=1e-20)
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
        Es.append(cas_energy)

        np.savez_compressed(
            savefile,
            labels=np.array(labels, dtype=object),
            Hs=np.array(Hs, dtype=object),
            casci_energies=np.array(Es),
        )

    # ---------------- operator pool (singles + doubles) ----------------
    # This matches the original FeS script: build excitation operators in the
    # active space, split into alpha/beta using even/odd spin-orbital indices.
    try:
        import sympy as sp
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `sympy`, required for FeS operator-pool generation. "
            "Install it (e.g., `pip install sympy`)."
        ) from e

    electrons = nelecas
    orbitals = ncas * 2

    n_alpha_sym, n_beta_sym = sp.symbols("N_alpha N_beta")
    eq1 = n_alpha_sym + n_beta_sym - electrons
    eq2 = n_alpha_sym - n_beta_sym - mol.spin

    soln = sp.solve((eq1, eq2), (n_alpha_sym, n_beta_sym), dict=True)
    n_alpha = int(soln[0][n_alpha_sym])
    n_beta = int(soln[0][n_beta_sym])

    alpha = [i for i in range(orbitals) if i % 2 == 0]
    beta = [i for i in range(orbitals) if i % 2 == 1]

    alpha_occ = alpha[:n_alpha]
    beta_occ = beta[:n_beta]
    alpha_virt = [a for a in alpha if a not in alpha_occ]
    beta_virt = [b for b in beta if b not in beta_occ]

    assert n_alpha + n_beta == electrons, "N_alpha + N_beta must equal nelecas"

    delta_sz = 0
    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles: list[list[int]] = []
    for r in alpha_occ:
        for p in alpha_virt:
            if sz[p] - sz[r] == delta_sz:
                singles.append([r, p])

    for r in beta_occ:
        for p in beta_virt:
            if sz[p] - sz[r] == delta_sz:
                singles.append([r, p])

    doubles = [
        sorted([s, r, q, p])
        for s in alpha_occ
        for r in beta_occ
        for q in alpha_virt
        for p in beta_virt
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    op1 = [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 = [
        qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})
        for x in doubles
    ]
    operator_pool = op1 + op2

    return H, cas_energy, operator_pool
