# Benchmark-QC

[![CI](https://github.com/AsthanaLab/BenchmarkQC/actions/workflows/python-package.yml/badge.svg)](https://github.com/AsthanaLab/BenchmarkQC/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-0A7BBB.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/datasets-78-6E56CF.svg)](datasets/catalog.json)

<p>
  <a href="docs/README.md">Documentation</a> ·
  <a href="datasets/catalog.json">Dataset catalog</a> ·
  <a href="docs/API.md">Python API</a> ·
  <a href="https://github.com/AsthanaLab/BenchmarkQC/issues">Report an issue</a>
</p>

Benchmark-QC is a versioned collection of molecular quantum-computing
benchmark Hamiltonians and portable active-space integral archives. It puts
the original BenchmarkQC systems and the corrected MolVQE-21 reference cases
in one consistent, checksum-pinned repository.

## What is included

The repository contains 78 benchmark catalog entries: 51 BenchmarkQC point
systems and 27 corrected MolVQE-21 point systems. Each bond distance and each
active-space/orbital variant is represented as its own catalog entry:

- `datasets/benchmarkQC/`: the C2, C2H4, CH2, Fe2S2, FeH, FeS, N2, O2, and U2
  benchmark families, including historical compatibility archives and newer
  numeric source archives.
- `datasets/molvqe21/`: 27 MolVQE-21 cases imported from Mushir’s corrected
  caches in `asthanaa/benchmarkkrylov`. The two source records without
  corrected benchmark caches are intentionally excluded.
- `src/benchmark_qc/`: importable loaders, integral conversion utilities, and
  physics-aware validation helpers.
- `datasets/catalog.json`: the machine-readable inventory with SHA-256 values,
  model metadata, and relative paths.
- `datasets/benchmarkQC/reference/reference_results.json`: common scalar CASCI, CISD,
  CCSD, Rényi-0.25, and cumulant records for all 51 BenchmarkQC points.
- `docs/SI_TABLE_I_AUDIT.md` and
  `datasets/benchmarkQC/reference/inventory.json`: the complete audit of all
  26 SI Table I geometry rows and their reconstruction provenance.
- `docs/ACTIVE_SPACE_AND_METHOD_AUDIT.md`: active-space evidence, limitations,
  literature checks, and scalar-method data coverage for all 51 BenchmarkQC
  records.
- `applications/`: a documented, ignored workspace for local experiments.
  Application calculations and their outputs are not part of this repository.

The checked-in dataset archives are reference artifacts, not a record of a
particular local environment. Dataset builders preserve their input hashes,
orbital selections, numerical conventions, and validation metadata so that a
future rebuild can be audited.

## Quick start

```sh
git clone https://github.com/AsthanaLab/BenchmarkQC.git
cd BenchmarkQC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
```

The package can then load the corrected MolVQE-21 cases directly:

```python
from benchmark_qc.molvqe21 import list_cases, load_hamiltonian, load_source_integrals

case = list_cases()[0]
hamiltonian = load_hamiltonian(case.case_id)
integrals = load_source_integrals(case.case_id)
print(case.case_id, integrals.n_qubits, hamiltonian.ref_energies[0])
```

For the complete set of loaders, file contracts, validation commands, and
dataset-specific caveats, start with [the documentation index](docs/README.md).

## Scientific scope and caveats

- The historical `.npz` Hamiltonian contract stores PennyLane operators in an
  object array. Load those files only from a trusted checkout and verify the
  catalog checksum first.
- Numeric integral archives use `allow_pickle=False` and are the preferred
  input for reproducible Hamiltonian reconstruction.
- Every catalog entry has a normalized spatial one-/two-electron integral
  archive. The 27 historical BenchmarkQC point archives that originally retained
  only JW operators now include a minimum-norm, real, chemist-symmetry
  reconstruction that regenerates the checked-in JW Hamiltonian. Those files
  are explicitly labeled as operator-equivalent frames; they do not claim to
  recover the unavailable original AO/MO coefficient frame.
- The accepted Fe2S2 historical records deliberately preserve their qualified
  nonstationary default-RHF/CASCI orbital frame. They are not natural-orbital
  or Active Space Finder selections; the constructed nested CAS(8e,8o) record
  is clearly labeled as a present-work control.
- MolVQE-21 contains only the 27 corrected-cache systems. `Ferrocene_ceo` and
  `feo4_2minus_ceo` are excluded because corrected benchmark caches were not
  available.
- Scalar CASCI, CISD, CCSD, Rényi-0.25, and two-body-cumulant records are
  available under `reference_results` for all 51 BenchmarkQC point systems.
  SI values are preserved as published; missing historical controls are computed
  from the checked-in normalized archives and marked with their provenance.
  Application calculations remain ignored under `applications/`.

## Contributing and citation

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before adding a dataset. New
systems must include a portable source archive, provenance, validation tests,
catalog metadata, and checksums. See [Adding a molecular dataset](docs/ADDING_DATASETS.md)
for the acceptance checklist.

If this repository supports published work, please use the metadata in
[CITATION.cff](CITATION.cff) and preserve the dataset commit in your methods
section.

Released under the [BSD 3-Clause license](LICENSE).
