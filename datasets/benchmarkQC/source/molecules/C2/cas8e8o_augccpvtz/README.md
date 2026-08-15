# C2 CAS(8e,8o)/aug-cc-pVTZ natural-orbital dataset

This directory contains the stretched neutral C2 benchmark at R = 2.2000
Angstrom in the X 1Sigma_g+ state. The 16-qubit Hamiltonian uses eight active
electrons in eight spatial orbitals and a Jordan-Wigner mapping with interleaved
alpha/beta spin orbitals.

The orbital basis is defined by diagonalizing the spin-summed one-particle
density matrix of the validated target-state CASCI solution and ordering the
orbitals by decreasing occupation. It is therefore a state-specific
CASCI-natural-orbital preconditioner, not a CASSCF or ordinary canonical-RHF
basis, and it uses exact target-state information.

Files:

- `casci_natural_orbitals/inputs/stretched_r2p2000.npz`: portable, pickle-free
  numeric integrals, orbital transformations, coefficients, occupations, and
  source hashes.
- `casci_natural_orbitals/C2_PES_H.npz`: backward-compatible BenchmarkQC
  archive with exactly `labels`, `Hs`, and `casci_energies`.
- `metadata.json`: physical definition, validation results, provenance, and
  checksums.

From the repository root, verify reconstruction and the physical sector with:

```sh
python datasets/benchmarkQC/source/molecules/C2/cas8e8o_augccpvtz/generate_hamiltonian.py --check
python datasets/benchmarkQC/source/molecules/C2/cas8e8o_augccpvtz/test_hamiltonian.py
```

At this stretched geometry, the lowest state in the unrestricted fixed-M_S=0
determinant sector is not the requested singlet. The validation therefore uses
an S^2=0 penalty to select the target root, then recomputes and reports its
energy and residual with the unpenalized physical Hamiltonian.

Rebuilding the portable input from the immutable QCANT artifacts requires both
the natural-orbital input and its parent active-HF-frame input:

```sh
python datasets/benchmarkQC/source/molecules/C2/cas8e8o_augccpvtz/build_source_archive.py \
  --natural-input <qcant-natural-input.npz> \
  --parent-input <qcant-parent-input.npz> \
  --output datasets/benchmarkQC/source/molecules/C2/cas8e8o_augccpvtz/casci_natural_orbitals/inputs/stretched_r2p2000.npz
```
