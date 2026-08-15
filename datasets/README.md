# Dataset store

All checked-in molecular reference data lives below this directory. The
top-level catalog is [`catalog.json`](catalog.json); every `data_path` and
`metadata_path` in it is relative to the repository root and carries a SHA-256
checksum where applicable.

## Families

- [`benchmarkQC/`](benchmarkQC/): 51 BenchmarkQC system payloads, including
  the reconstructed SI Table I geometry cases.
- [`molvqe21/`](molvqe21/): 27 corrected MolVQE-21 system payloads in the same
  Hamiltonian and integral conventions.
- Both families use the same per-system contract:
  `systems/<case_id>/hamiltonian.npz`,
  `systems/<case_id>/inputs/source_integrals.npz`, and
  `systems/<case_id>/metadata.json`.
- Every catalog entry has a `data_path` Hamiltonian and an
  `integral_data_path` normalized spatial integral archive. Historical
  Hamiltonian-only BenchmarkQC cases use a documented JW-equivalent
  reconstruction when their original orbital coefficients were not retained.
- [`benchmarkQC/si_table_i_inventory.json`](benchmarkQC/si_table_i_inventory.json):
  validation and provenance for every SI Table I geometry row.

Dataset builders, source manifests, acceptance reports, and validation scripts
stay next to the family they document. Application calculations do not belong
here; use [`../applications/`](../applications/) for ignored local outputs.
