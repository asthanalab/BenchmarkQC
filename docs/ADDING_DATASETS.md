# Adding a molecular dataset

Use one folder per chemical system under the relevant family in `datasets/`.
If a system has more than one
basis, active space, state, or orbital definition, keep the variants in clearly
named subfolders rather than overwriting an earlier archive.

## Required contribution

A new dataset should provide:

1. A reproducible generator script or notebook with geometry, units, charge,
   spin, basis, active-space definition, and orbital-selection procedure.
2. A Hamiltonian archive using the exact `labels`, `Hs`, and
   `casci_energies` contract documented in `NPZ_FORMAT.md`.
3. A validation command that compares every stored reference in the correct
   electron/spin sector.
4. Metadata recording the physical model, mapping and wire order, software
   versions, provenance, limitations, and SHA-256 checksums.
5. Numeric source integrals, geometries, orbital coefficients, and
   active-orbital indices whenever redistribution is permitted.
6. A catalog entry in `datasets/catalog.json` and a short row in `DATASETS.md`.
7. Automated tests for file integrity, schema, term counts/qubit count, source
   reconstruction, and reference energies.

Do not add generated caches, local paths, credentials, hostnames, job logs, or
unreviewed intermediate checkpoints. Do not create an empty or synthetic
Hamiltonian merely to make a system appear complete.

## Recommended layout

```text
datasets/benchmarkQC/Molecule/
  README.md
  model/
    README.md
    metadata.json
    generate_hamiltonians.py
    test_hamiltonians.py
    orbital_or_method_variant/
      inputs/
        geometry_id.npz
      Molecule_PES_H.npz
```

The final variant level may be omitted when a model has only one orbital/method
definition. Legacy folders may retain their original names. New contributions
should be self-contained, use relative paths, and leave existing datasets
byte-for-byte unchanged unless a separate, documented correction is required.
