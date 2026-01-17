"""N2 Hamiltonian generation helpers."""

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
    """Generate a qubit Hamiltonian H and CASCI energy for N2 (active-space selected by ASF).

    This mirrors the original `N2/ham_pyscf.py` implementation.
    """

    # Imports that are optional for just importing the package.
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
            "Missing dependency `asf` (active space finder). Install it to generate N2 Hamiltonians."
        ) from e

    basis = bse.get_basis(basis_input, elements=elements, fmt="nwchem")

    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    mol.verbose = 0

    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton(mf).run()

    with suppress_stdout():
        active_space = find_from_mol(mol, max_norb=ncas, min_norb=3, verbose=0)

    mycas = CASCI(mol, ncas=active_space.norb, nelecas=active_space.nel)
    mo_guess = mycas.sort_mo(active_space.mo_list, active_space.mo_coeff, base=0)

    mycas.kernel(mo_coeff=mo_guess, verbose=0)
    cas_energy = mycas.kernel()[0]

    one_mo, ecore = mycas.get_h1eff(mycas.mo_coeff)
    h2ecas = mycas.get_h2eff(mycas.mo_coeff)

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

    return H, cas_energy
