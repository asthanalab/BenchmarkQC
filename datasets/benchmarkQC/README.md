# BenchmarkQC systems

This folder contains the original BenchmarkQC molecular systems and the
reconstructed SI Table I cases in a common dataset namespace. Each geometry
and active-space/orbital variant has its own catalog record and system
directory. All 51 published cases use the same layout:

```text
benchmarkQC/
├── systems/<case_id>/
│   ├── hamiltonian.npz
│   ├── inputs/source_integrals.npz
│   └── metadata.json
├── metadata.json
├── reference_results.json
└── si_table_i_inventory.json
```

The molecule-specific folders retain source archives, provenance, builders,
and validation material. They are not alternate locations for the runnable
system payloads; the catalog and loaders always use `systems/<case_id>/`.

| System | Contents |
| --- | --- |
| `datasets/benchmarkQC/systems/` | all 51 runnable system payloads in the canonical layout |
| `datasets/benchmarkQC/C2/` | aug-cc-pVTZ CAS(8e,8o) source and validation material |
| `datasets/benchmarkQC/C2H4/` | aug-cc-pVDZ CAS(2e,2o), planar and twisted geometries |
| `datasets/benchmarkQC/CH2/` | aug-cc-pVTZ CAS(6e,6o), bent/open geometries |
| `datasets/benchmarkQC/Fe2S2/` | CAS(10e,10o) SI geometries plus accepted historical/control variants |
| `datasets/benchmarkQC/FeH/` | aug-cc-pVTZ-DK/SF-X2C CAS(9e,10o) geometries |
| `datasets/benchmarkQC/FeS/` | historical ANO-RCC-MB CAS(6e,6o) scan plus SI CAS(14e,10o) cases |
| `datasets/benchmarkQC/N2/` | historical STO-6G scan plus cc-pVDZ CAS(10e,8o) variants |
| `datasets/benchmarkQC/O2/` | aug-cc-pVTZ CAS(12e,8o) geometries |
| `datasets/benchmarkQC/U2/` | historical CAS(6e,6o) points plus SI CAS(6e,10o) certificate cases |

The root catalog at [`../catalog.json`](../catalog.json) is the authoritative
list of published archives. See [the full dataset inventory](../../docs/DATASETS.md)
for geometry, active-space, provenance, and acceptance details.

The complete search of the SI Table I sources and the older OneDrive artifact
store is documented in [the SI Table I payload audit](../../docs/SI_TABLE_I_AUDIT.md).
All 26 SI Table I geometry rows have a validated payload; the inventory records
which U₂ rows were reconstructed from validated QCANT certificate inputs rather
than copied from dataless OneDrive placeholders.
