import openfermion
import numpy as np
import copy as cp
print('---SRJ---U2-SD-Comb-3times-----------')
print('------SRJ----2.4-SD-Comb---------')
print('================SRJ====N2_2.50A_FermionicSD_3cycles======')
print('==============SRJ=====[2Fe-2S] (6,6)======================')

import numpy as np
from pyscf import gto, scf, mcscf, fci, tools

# --- 1. Read FCIDUMP and set up fake HF as you already do ---
fcidump = tools.fcidump.read('fe2s2.txt')

h1_ao = fcidump['H1']
h2_ao = fcidump['H2']
nelec_tot = fcidump['NELEC']
norb  = h1_ao.shape[0]
ecore = fcidump.get('ECORE', 0.0)

mol = gto.Mole()
mol.nelectron = nelec_tot
mol.incore_anyway = True
mol.build()

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1_ao
mf.get_ovlp  = lambda *args: np.eye(norb)
mf._eri = h2_ao
mf.energy_nuc = lambda *args: ecore
mf.kernel()

# --- 2. Define CAS(6,6) and extract active-space integrals ---
ncas = 6
nelec_cas = (3, 3)  # singlet CAS(6,6)

mycas = mcscf.CASCI(mf, ncas, sum(nelec_cas))
mycas.fix_spin(ss=0.0)

# choose your 6 active MOs; for now, just take some window
# (you'll probably want to choose these more carefully)
# Example: last 6 MOs:
#cas_list = list(range(norb - ncas, norb))
#mycas = mycas.set_active_space(cas_list)

h1e_cas, ecore_cas = mycas.get_h1eff(mycas.mo_coeff)
h2e_cas = mycas.get_h2eff(mycas.mo_coeff)

# h1e_cas, h2e_cas are your CAS(6,6) one- and two-electron integrals
# dimensions: h1e_cas: (6,6), h2e_cas: (6,6,6,6)

# --- 3. Run FCI on the CAS Hamiltonian ---

cisolver = fci.direct_spin1.FCI()
e_cas, fcivec = cisolver.kernel(h1e_cas, h2e_cas, ncas, nelec_cas)

etot = e_cas + ecore_cas
print("CAS(6,6) FCI energy =", etot)


import pyscf
print('H1 eff Ham', h1e_cas.shape)
#print('H2 eff Hamiltonian', h2ecas.shape)


two_mo = pyscf.ao2mo.restore('1', h2e_cas, norb=6)
print('--------------------Converting to physcist notation--------------')
#--------------------------Converting to physcist notation----------------------------
two_mo = np.swapaxes(two_mo, 1, 3)


print('two_mo shape', two_mo.shape)

print('Core constant', ecore_cas)

import pennylane as qml
one_mo = h1e_cas
#two_mo = h2ecas
core_constant = np.array([ecore_cas])

H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo, cutoff=1e-20)

#print(H_fermionic)

H = qml.jordan_wigner(H_fermionic)

import openfermion
import numpy as np
import copy as cp

import pennylane as qml

from openfermion import FermionOperator

from openfermion import *

def generate_SQ_Operators():
    """
    0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
    """

    print(" Form singlet SD operators")
    fermi_ops = []


    n_occ = 3
    n_vir = 3

    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for a in range(0,n_vir):
            aa = 2*n_occ + 2*a
            ab = 2*n_occ + 2*a+1


            print('ia-Occ alpha', ia)
            print('ib-Occ Beta', ib)
            print('aa - Virt alpha', aa)
            print('ab - Virt Beta', ab)
            termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))

            termA -= hermitian_conjugated(termA)

            termA = normal_ordered(termA)
            #print('Term A after normal_ordered', termA)
            #Normalize
            coeffA = 0
            for t in termA.terms:
                coeff_t = termA.terms[t]
                coeffA += coeff_t * coeff_t

            if termA.many_body_order() > 0:
                termA = termA/np.sqrt(coeffA)
                fermi_ops.append(termA)


    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,n_occ):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                for b in range(a,n_vir):
                    ba = 2*n_occ + 2*b
                    bb = 2*n_occ + 2*b+1

                    termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))

                    termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                    termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                    termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                    termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)

                    termA -= hermitian_conjugated(termA)
                    termB -= hermitian_conjugated(termB)

                    termA = normal_ordered(termA)
                    termB = normal_ordered(termB)

                    #Normalize
                    coeffA = 0
                    coeffB = 0
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
                    for t in termB.terms:
                        coeff_t = termB.terms[t]
                        coeffB += coeff_t * coeff_t


                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        fermi_ops.append(termA)

                    if termB.many_body_order() > 0:
                        termB = termB/np.sqrt(coeffB)
                        fermi_ops.append(termB)

    n_ops = len(fermi_ops)
    print(" Number of operators: ", n_ops)
    #print('The operators are', fermi_ops)
    return fermi_ops
# }}}

fermi_ops = generate_SQ_Operators()
x = [None] * len(fermi_ops)
print('Before loop, len of x', len(x))

for i in range(len(fermi_ops)):
    x[i] = qml.from_openfermion(fermi_ops[i])
print('Total operators after loop  are', len(x))



import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from itertools import chain
import itertools
import time
import re
import scipy
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
ash_excitation = []
energies = []
excitations= []
old_grad = []
excitationlist = []
generatingfns = []
gs_energy = []
grad_GCIM = []  # To store the highest gradient excitation values
operator_check = []  # To store the highest gradient excitation operators
theta = np.pi/4
print('Theta is', theta)
X = qml.PauliX
Y = qml.PauliY
Z = qml.PauliZ
I = qml.Identity

electrons = 6
qubits = 12

# assume: H, x, theta, gs_energy, grad_GCIM are defined globally

dev = qml.device("lightning.qubit", wires=qubits)

@qml.qnode(dev)
def hf_stateprep(wires):
    target_state = np.zeros(2**qubits)
    target_state[4032] = 1.0
    qml.StatePrep(target_state, wires=range(qubits))
    return qml.state()

hf_state = hf_stateprep(wires=qubits)

dev_meas1 = qml.device("lightning.qubit", wires=qubits, shots=10_000_000)
@qml.qnode(dev_meas1)
def measure(ostate):
    qml.StatePrep(ostate, wires=range(qubits))
    return qml.counts()

ref_state = hf_state      # starting reference
#ash_excitation = []       # global history of chosen operators (optional)
#Def. of operator pool



def adaptvqe(ref_state, adapt_it, e_th=1e-12):
    """One GCIM+ADAPT run:
       - starts from ref_state
       - uses each operator at most once in this run
    """
    energies = []
    excitations = []
    operator_check = []    # <‑‑ resets every run
    ash_excitation = []

    dev_loc = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev_loc)
    def circuit(state, H):
        qml.StatePrep(state, wires=range(qubits))
        return qml.expval(H)

    @qml.qnode(dev_loc)
    def commutator_0(H, w, k):
        qml.StatePrep(k, wires=range(qubits))
        res = qml.commutator(H, w)
        return qml.expval(res)

    @qml.qnode(dev_loc)
    def commutator_1(H, w, k):
        qml.StatePrep(k, wires=range(qubits))
        res = qml.commutator(H, w)
        return qml.expval(res)

    @qml.qnode(dev_loc)
    def new_state(ref_state, ash_excitation):
        qml.StatePrep(ref_state, wires=range(qubits))
        for op_idx in ash_excitation:
            s = qml.jordan_wigner(x[op_idx])
            qml.exp(s * theta / 2, num_steps=1)
        return qml.state()

    @qml.qnode(dev_loc)
    def ind_state(ref_state, op_idx):
        qml.StatePrep(ref_state, wires=range(qubits))
        t = qml.jordan_wigner(x[op_idx])
        qml.exp(t * theta / 2, num_steps=1)
        return qml.state()

    print("Ref energy:", circuit(ref_state, H))
    print("Ref measurement:", measure(ref_state))

    states = [ref_state]
    max_operator = None

    for j in range(1, adapt_it + 1):
        print("ADAPT step", j, flush=True)
        max_value = float("-inf")
        k = states[-1] if states else hf_state

        # scan full pool, but skip operators already used in THIS run
        for i in range(len(x)):
            if str(i) in operator_check:
                continue
            w = qml.fermi.jordan_wigner(x[i])
            if np.array_equal(k, hf_state):
                val = abs(2 * commutator_0(H, w, k))
            else:
                val = abs(2 * commutator_1(H, w, k))

            if val > max_value:
                max_value = val
                max_operator = i

        if max_operator is None:
            print("No operator left in pool for this run.")
            break

        print("  chosen op:", max_operator, "grad:", max_value)
        operator_check.append(str(max_operator))
        ash_excitation.append(max_operator)       # global history if you want it

        # build new ansatz state (all chosen ops on ref_state)
        ostate = new_state(ref_state, ash_excitation)

        # optional individual excitation state
        if j >= 2:
            states.append(ind_state(ref_state, max_operator))

        states.append(ostate)

        # GCIM generalized eigenproblem in span(states)
        M = np.zeros((len(states), len(states)), dtype=complex)
        S = np.zeros_like(M)
        Ham_matrix = qml.matrix(H, wire_order=range(qubits))

        for a in range(len(states)):
            for b in range(len(states)):
                left = states[a].T.conj()
                right = states[b]
                M[a, b] = left @ (Ham_matrix @ right)
                S[a, b] = left @ right

        eps = 1e-10
        S_reg = S + eps * np.eye(S.shape[0])
        eig, evec = scipy.linalg.eigh(M.real, S_reg.real)

        gs_energy.append(eig[0])
        grad_GCIM.append(max_value)
        print("  GS energy list:", gs_energy, flush=True)

    # build GCIM ground state for this run
    final_state = sum(coeff * state for coeff, state in zip(evec[:, 0], states))
    final_state = final_state / np.linalg.norm(final_state)
    return ash_excitation, states, eig, gs_energy, Ham_matrix, hf_state, max_operator, evec, ref_state, grad_GCIM, final_state



for run in range(3):   # Outer loop over GCIM runs
    print("=== GCIM run", run, "===")
    ash_excitation, states, eig, gs_energy, Ham_matrix, hf_state, max_op, evec, ref_state, grad_GCIM, final_state = adaptvqe(
        ref_state,
        adapt_it=len(x)
    )
    print('ash_excitation', ash_excitation)
    # Save the state with a unique filename for this run
    filename = f'vi66_GCIM{run}.npy'
    np.save(filename, final_state)
    print(f"Saved final state for run {run} to {filename}")

    print("Measurement of final_state:", measure(final_state))
    ref_state = final_state    # improved reference for the next run
    print('\n')
