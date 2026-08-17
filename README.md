# Benchmark-QC

<p align="center">
  <strong>Portable, checksum-pinned molecular benchmark Hamiltonians for quantum chemistry.</strong><br>
  Reproducible active-space datasets, normalized integral archives, and Python loaders for quantum-computing research.
</p>

<p align="center">
  <a href="https://github.com/AsthanaLab/BenchmarkQC/actions/workflows/python-package.yml"><img src="https://github.com/AsthanaLab/BenchmarkQC/actions/workflows/python-package.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/benchmark-qc/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-0A7BBB.svg" alt="BSD 3-Clause license"></a>
  <a href="datasets/catalog.json"><img src="https://img.shields.io/badge/dataset%20entries-78-6E56CF.svg" alt="78 dataset entries"></a>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> ·
  <a href="datasets/catalog.json">Dataset catalog</a> ·
  <a href="docs/API.md">Python API</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="https://github.com/AsthanaLab/BenchmarkQC/issues">Report an issue</a>
</p>

Benchmark-QC is a versioned collection of molecular quantum-computing benchmark Hamiltonians and portable active-space integral archives. It brings the original BenchmarkQC systems and the corrected MolVQE-21 reference cases into one consistent, checksum-pinned repository.

## At a glance

| Collection | Entries | What it contains |
| --- | ---: | --- |
| **BenchmarkQC** | 51 | C2, C2H4, CH2, Fe2S2, FeH, FeS, N2, O2, and U2 benchmark systems |
| **Corrected MolVQE-21** | 27 | Al₃⁻, BeH₂, CH₂O, CO, cyclobutadiene, FeNO²⁺, Fe(NO)(CO)₃⁻, FeO, H₄–H₁₀ chains, hexatriene, m-benzyne, magnesium porphyrin, and N₂ |
| **Total catalog** | **78** | Point-level entries with paths, metadata, and SHA-256 checksums |

Each catalog entry has a runnable Hamiltonian archive, a normalized spatial integral archive, per-system metadata, and provenance links. The catalog is the authoritative index for the checked-in data.

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

Load a corrected MolVQE-21 case directly from the package:

```python
from benchmark_qc.molvqe21 import list_cases, load_hamiltonian, load_source_integrals

case = list_cases()[0]
hamiltonian = load_hamiltonian(case.case_id)
integrals = load_source_integrals(case.case_id)

print(case.case_id)
print(integrals.n_qubits, hamiltonian.ref_energies[0])
```

For installation from PyPI, use:

```sh
python -m pip install benchmark-qc
```

## Repository map

```text
BenchmarkQC/
├── src/benchmark_qc/                 # Importable loaders and validation utilities
├── datasets/
│   ├── catalog.json                  # Authoritative 78-entry dataset index
│   ├── benchmarkQC/
│   │   ├── systems/                  # 51 runnable BenchmarkQC point systems
│   │   ├── reference/                # Scalar reference results and inventories
│   │   └── source/                   # Shared manifest, provenance, and builders
│   └── molvqe21/
│       ├── systems/                  # 27 corrected MolVQE-21 point systems
│       ├── reference/                # Reference results and inventories
│       └── source/                   # Manifests, overrides, and dataset builders
├── docs/                             # Usage, API, formats, audits, and validation
├── applications/                     # Ignored workspace for local experiments
├── tests/                            # Schema, physics, and reproducibility tests
├── tools/                            # Dataset maintenance and audit utilities
├── CITATION.cff                      # Machine-readable citation metadata
└── pyproject.toml                    # Package metadata and dependencies
```

## Documentation

Start with the [documentation index](docs/README.md) for the complete guide:

- [Repository layout](docs/REPOSITORY_LAYOUT.md)
- [Usage examples](docs/USAGE.md)
- [Dataset catalog](docs/DATASETS.md)
- [Package API](docs/API.md)
- [Integral archives and reconstruction](docs/INTEGRAL_ARCHIVES.md)
- [Saved NPZ format](docs/NPZ_FORMAT.md)
- [Validation and release checks](docs/VALIDATION.md)
- [Adding a molecular dataset](docs/ADDING_DATASETS.md)
- [Active-space and method-data audit](docs/ACTIVE_SPACE_AND_METHOD_AUDIT.md)

## Validation

Run the complete local validation gate from the repository root:

```sh
python -m pip install -e .
python -m pytest -q
python -m build
```

The GitHub Actions workflow runs the test suite on Python 3.11 and 3.12, checks for syntax and undefined-name errors, and builds the source distribution and wheel.

## Citation

If you use the BenchmarkQC collection, cite [*Chemically decisive benchmarks on the path to quantum utility*](https://ui.adsabs.harvard.edu/abs/2026arXiv260110813P/abstract). If you use the corrected MolVQE-21 collection, cite [*Exponential Scaling Barriers for Variational Quantum Eigensolvers*](https://arxiv.org/abs/2603.13073), which introduces the MolVQE-21 benchmark set.

```bibtex
@misc{Sundar2026ChemicallyDecisiveBenchmarks,
  title         = {Chemically decisive benchmarks on the path to quantum utility},
  author        = {Sundar, Srivathsan Poyyapakkam and Abraham, Vibin and Peng, Bo and Asthana, Ayush},
  year          = {2026},
  eprint        = {2601.10813},
  archivePrefix = {arXiv},
  primaryClass  = {physics.chem-ph},
  doi           = {10.48550/arXiv.2601.10813}
}

@misc{Hagelueken2026ExponentialScalingBarriers,
  title         = {Exponential Scaling Barriers for Variational Quantum Eigensolvers},
  author        = {Hagelueken, Manuel and Kreplin, David A. and Wieland, Florian and Huber, Marco F. and Roth, Marco},
  year          = {2026},
  eprint        = {2603.13073},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2603.13073}
}
```

Please also see [CITATION.cff](CITATION.cff) for machine-readable repository metadata. When reporting results, preserve the dataset commit or release used in the methods section.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before adding or changing a dataset. New systems should include portable source data, provenance, validation tests, catalog metadata, and checksums. The [dataset acceptance checklist](docs/ADDING_DATASETS.md) describes the required files and review steps.

## License

Released under the [BSD 3-Clause license](LICENSE).
