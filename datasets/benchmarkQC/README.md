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
├── reference/
│   ├── reference_results.json
│   └── inventory.json
├── source/
│   ├── manifest.csv
│   └── molecules/<molecule>/
├── metadata.json
└── README.md
```

The molecule-specific source folders are kept under `source/molecules/` so the
family root has the same organization as MolVQE-21. They contain provenance,
builders, and validation material only; the catalog and loaders always use
`systems/<case_id>/` for runnable system payloads.

| System | Contents |
| --- | --- |
| `datasets/benchmarkQC/systems/` | all 51 runnable system payloads in the canonical layout |
| `datasets/benchmarkQC/source/molecules/C2/` | aug-cc-pVTZ CAS(8e,8o) source and validation material |
| `datasets/benchmarkQC/source/molecules/C2H4/` | aug-cc-pVDZ CAS(2e,2o), planar and twisted geometries |
| `datasets/benchmarkQC/source/molecules/CH2/` | aug-cc-pVTZ CAS(6e,6o), bent/open geometries |
| `datasets/benchmarkQC/source/molecules/Fe2S2/` | CAS(10e,10o) SI geometries plus accepted historical/control variants |
| `datasets/benchmarkQC/source/molecules/FeH/` | aug-cc-pVTZ-DK/SF-X2C CAS(9e,10o) geometries |
| `datasets/benchmarkQC/source/molecules/FeS/` | historical ANO-RCC-MB CAS(6e,6o) scan plus SI CAS(14e,10o) cases |
| `datasets/benchmarkQC/source/molecules/N2/` | historical STO-6G scan plus cc-pVDZ CAS(10e,8o) variants |
| `datasets/benchmarkQC/source/molecules/O2/` | aug-cc-pVTZ CAS(12e,8o) geometries |
| `datasets/benchmarkQC/source/molecules/U2/` | historical CAS(6e,6o) points plus SI CAS(6e,10o) certificate cases |

The root catalog at [`../catalog.json`](../catalog.json) is the authoritative
list of published archives. The shared family manifest is
[`source/manifest.csv`](source/manifest.csv), and the shared reference indexes
are in [`reference/`](reference/). See [the full dataset inventory](../../docs/DATASETS.md)
for geometry, active-space, provenance, and acceptance details.

The complete search of the SI Table I sources and the older OneDrive artifact
store is documented in [the SI Table I payload audit](../../docs/SI_TABLE_I_AUDIT.md).
All 26 SI Table I geometry rows have a validated payload; the inventory records
which U₂ rows were reconstructed from validated QCANT certificate inputs rather
than copied from dataless OneDrive placeholders.
