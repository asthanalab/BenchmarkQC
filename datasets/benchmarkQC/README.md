# BenchmarkQC systems

This folder contains the original BenchmarkQC molecular systems and the
reconstructed SI Table I cases in a common dataset namespace. Each geometry
and active-space/orbital variant has its own catalog record and system
directory. The directory keeps saved reference archives, portable metadata,
compatibility scripts, and dataset-specific validation close together.

| System | Contents |
| --- | --- |
| `datasets/benchmarkQC/C2/` | aug-cc-pVTZ CAS(8e,8o) reference and SI equilibrium case |
| `datasets/benchmarkQC/C2H4/` | aug-cc-pVDZ CAS(2e,2o), planar and twisted geometries |
| `datasets/benchmarkQC/CH2/` | aug-cc-pVTZ CAS(6e,6o), bent/open geometries |
| `datasets/benchmarkQC/Fe2S2/` | CAS(10e,10o) SI geometries plus accepted historical/control variants |
| `datasets/benchmarkQC/FeH/` | aug-cc-pVTZ-DK/SF-X2C CAS(9e,10o) geometries |
| `datasets/benchmarkQC/FeS/` | legacy ANO-RCC-MB CAS(6e,6o) scan plus SI CAS(14e,10o) cases |
| `datasets/benchmarkQC/N2/` | legacy STO-6G scan plus cc-pVDZ CAS(10e,8o) variants |
| `datasets/benchmarkQC/O2/` | aug-cc-pVTZ CAS(12e,8o) geometries |
| `datasets/benchmarkQC/U2/` | legacy CAS(6e,6o) points plus SI CAS(6e,10o) certificate cases |

The root catalog at [`../catalog.json`](../catalog.json) is the authoritative
list of published archives. See [the full dataset inventory](../../docs/DATASETS.md)
for geometry, active-space, provenance, and acceptance details.

The complete search of the SI Table I sources and the older OneDrive artifact
store is documented in [the SI Table I payload audit](../../docs/SI_TABLE_I_AUDIT.md).
All 26 SI Table I geometry rows have a validated payload; the inventory records
which U₂ rows were reconstructed from validated QCANT certificate inputs rather
than copied from dataless OneDrive placeholders.
