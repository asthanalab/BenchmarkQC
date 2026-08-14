# Usage

## Installation (PyPI)

To install the package from PyPI:

```sh
pip install benchmark-qc
```

This repo contains benchmark qubit Hamiltonians for:

- C2 (folder: `datasets/benchmarkQC/C2/`)
- C2H4 (folder: `datasets/benchmarkQC/C2H4/`)
- CH2 (folder: `datasets/benchmarkQC/CH2/`)
- FeH (folder: `datasets/benchmarkQC/FeH/`)
- N2 (folder: `datasets/benchmarkQC/N2/`)
- O2 (folder: `datasets/benchmarkQC/O2/`)
- FeS (folder: `datasets/benchmarkQC/FeS/`)
- Fe2S2 (folder: `datasets/benchmarkQC/Fe2S2/`)
- U2 (folder: `datasets/benchmarkQC/U2/`)

The system folders contain saved `.npz` Hamiltonians and validation metadata.
Legacy systems retain their input notebooks; current datasets additionally
provide portable numeric source archives and model-level metadata. Every
catalog record is a single geometry/active-space point, even when a legacy
multi-point compatibility archive is also retained.

## Quickstart (recommended)

Create and use a dedicated conda environment:

- `conda create -n bench python=3.11`
- `conda install -n bench -c conda-forge numpy scipy pyscf sympy basis_set_exchange`
- `conda run -n bench python -m pip install pennylane`
- `conda run -n bench python -m pip install -e .`

## Run Hamiltonian sanity checks

From the repo root:

- `conda run -n bench python datasets/benchmarkQC/N2/test_n2_hamiltonian.py --index 0`
- `conda run -n bench python datasets/benchmarkQC/FeS/test_fes_hamiltonian.py --index 0`
- `conda run -n bench python datasets/benchmarkQC/U2/test_u2_hamiltonian.py --index 0`

For the current N2 CAS(10e,8o)/cc-pVDZ calculation:

- `conda run -n bench python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/test_hamiltonians.py --variant canonical --index 0`
- `conda run -n bench python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/test_hamiltonians.py --variant casci_natural_orbitals --index 1`
- `conda run -n bench python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/generate_hamiltonians.py --check`

For the C2 CAS(8e,8o)/aug-cc-pVTZ natural-orbital calculation:

- `conda run -n bench python datasets/benchmarkQC/C2/cas8e8o_augccpvtz/generate_hamiltonian.py --check`
- `conda run -n bench python datasets/benchmarkQC/C2/cas8e8o_augccpvtz/test_hamiltonian.py`

For an accepted Fe2S2 dataset, use the package loader with one of
`historical_cas4e4o`, `historical_cas6e6o`, `historical_cas8e6o`, or the
distinct present-work control `nested_cas8e8o`:

```python
from benchmark_qc import fe2s2

hamiltonian = fe2s2.load_hamiltonian("historical_cas6e6o")
source = fe2s2.load_source_integrals("historical_cas6e6o")
```

You can also choose a point by bond length (nearest stored label):

- `conda run -n bench python datasets/benchmarkQC/N2/test_n2_hamiltonian.py --bond 1.4`
- `conda run -n bench python datasets/benchmarkQC/FeS/test_fes_hamiltonian.py --bond 2.4`
- `conda run -n bench python datasets/benchmarkQC/U2/test_u2_hamiltonian.py --bond 2.48`

## Stored geometry point labels

In each saved `.npz`, the `labels` array stores the geometry “point label” used at
generation time. For these diatomics, it is the bond length in **Angstrom** (PySCF
default unit, since the generators do not override `mol.unit`).

- **N2** (`datasets/benchmarkQC/N2/N2_PES_H1.npz`, 11 points):
	- $R$ (Å) = 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
- **N2 CAS(10e,8o)/cc-pVDZ**, canonical and CASCI-natural-orbital variants (2 points each):
	- $R$ (Å) = 1.0977, 2.0000
- **C2 CAS(8e,8o)/aug-cc-pVTZ**, state-specific CASCI natural orbitals (1 point):
	- $R$ (Å) = 2.2000
- **FeS** (`datasets/benchmarkQC/FeS/FeS_PES_H.npz`, 14 points):
	- $R$ (Å) = 1.826, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 3.7, 4.0, 4.2, 4.5, 4.8
- **Fe2S2 historical CAS(4e,4o), CAS(6e,6o), and CAS(8e,6o)** (1 point each):
	- label = 0.0 (archive point identifier, not a bond length; the parent FCIDUMP does not encode coordinates)
- **Fe2S2 constructed nested CAS(8e,8o)** (1 point):
	- label = 0.0 (the same archived point identifier; this control activates parent MOs 11--18 and is not a historical paper active space)
- **U2** (`datasets/benchmarkQC/U2/U2_PES_H.npz`, 2 points):
	- $R$ (Å) = 2.4, 2.48
- **SI Table I reconstructed cases**:
	- C2H4: planar 0° and twisted 90° CAS(2e,2o)
	- CH2: 102° and 140° CAS(6e,6o)
	- FeH: 1.5680 and 2.3000 Å CAS(9e,10o)
	- FeS: 2.0170 and 3.0000 Å CAS(14e,10o)
	- Fe2S2: published and bridge-stretched CAS(10e,10o)
	- O2: 1.2075 and 2.0000 Å CAS(12e,8o)
	- U2: 2.4300 and 2.8000 Å CAS(6e,10o)

## Important: historical Fe2S2 orbital frame

The accepted historical Fe2S2 variants faithfully replay the paper's
default-RHF/CASCI partitions.  The accepted `nested_cas8e8o` control applies
the same recovered partition rule to parent MOs 11--18, so it strictly
contains CAS(6e,6o), but it is not one of the paper's historical spaces.  The
shared RHF calculation stopped before orbital stationarity, and every record
discloses that limitation. They are not natural-orbital or Active Space Finder
selections. A modern DMRG/Active Space Finder CAS(6e,6o) workflow is retained
separately but has not been accepted into `datasets/catalog.json`.

## Important: FeS is open-shell/high-spin

FeS is an open-shell system. The saved reference energy is for a *specific* electron/spin sector.

The FeS test script compares energies in that same sector:
- `--nelec` = active-space electrons (default `6`, matches the FeS notebook)
- `--spin`  = PySCF convention `spin = 2S` (default `4`, matches the FeS notebook)

If you change the physical state you want to target (different `spin`), the stored `casci_energies`
will only match if the `.npz` was generated with that same choice.

## Generating Hamiltonians (not required for tests)

The Hamiltonian-generation functions are exposed for notebook compatibility:

- `datasets/benchmarkQC/N2/ham_pyscf.py` exports `H_gen`
- `datasets/benchmarkQC/FeS/FeS_pyscf.py` exports `H_gen`
- `datasets/benchmarkQC/U2/U2_ham1.py` exports `build_u2_reference` and `H_gen`

The actual implementations live in the package:
- `benchmark_qc.n2.H_gen`
- `benchmark_qc.fes.H_gen`
- `benchmark_qc.u2.build_u2_reference`, `benchmark_qc.u2.H_gen`

Some generation code depends on an external active-space finder (`asf`). The test scripts do not.

The new N2 generator does not rerun SCF or CASCI. It deterministically maps the
checked-in, remote-generated spatial integrals to PennyLane Jordan-Wigner terms.
The C2 generator follows the same deterministic mapping. Its target-state test
uses an S^2=0 penalty because the unrestricted fixed-M_S=0 ground state is not
the requested singlet, then evaluates the unpenalized physical Hamiltonian.
