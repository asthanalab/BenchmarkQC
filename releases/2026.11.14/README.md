# BenchmarkQC supporting dataset

Supporting data for *Strongly correlated chemical test systems for quantum algorithms*.

- [Download the dataset](BenchmarkQC-2026.11.14-data.zip)
- [Zenodo deposit: 10.5281/zenodo.24237572](https://doi.org/10.5281/zenodo.24237572)
- [SHA-256 checksum](SHA256SUMS)

The archive covers all 18 models in Supporting Information Table I and an additional C2 CASCI natural-orbital representation. Its 19 model directories contain 68 geometry or integral-frame records across nine molecular families.

Each record contains a geometry or integral-frame definition, numerical Pauli Hamiltonian, spatial integrals, reference results, state metadata and an array inventory. Available orbital coefficients and construction records are stored with the corresponding model or point. The archive includes format documentation, the SI reference table, citation metadata, a license and file-integrity checksums.

N2 CAS(4e,4o)/STO-6G at 1.20 and 2.00 angstrom includes full orbital coefficients, construction settings and numerical validation records. The 2.00 angstrom record includes the initial UHF density. Orbital metadata distinguish molecular orbital coefficients from model-basis identity matrices.

This archive is identical to the file distributed through the Zenodo deposit. Its internal `catalog.json` indexes the publication dataset. The repository-level `datasets/catalog.json` indexes the runnable datasets supported by the Python loaders.
