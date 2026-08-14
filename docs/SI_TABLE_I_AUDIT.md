# SI Table I payload audit

This audit records the search of both OneDrive project trees named in the
working notes:

- `code/qcomputing/U2response`
- `code/qcomputing/Benchmark-QC`

The authoritative row inventory is
[`datasets/benchmarkQC/si_table_i_inventory.json`](../datasets/benchmarkQC/si_table_i_inventory.json).
It covers all 13 SI Table I molecule/active-space variants and all 26 geometry
rows. Every row now has a validated, portable Hamiltonian payload. SI result
JSON, plots, checkpoints, and application logs are not benchmark inputs.

## Findings

| Availability | SI rows | Meaning |
| --- | ---: | --- |
| Checked-in Hamiltonian payload | 26 | All SI Table I geometry rows, including every active-space/orbital variant |
| Cloud-placeholder Hamiltonian payload | 0 | The original expanded U₂ SI-side files remain dataless placeholders, so their validated QCANT certificate inputs were used instead |
| Results/metadata only | 0 | No SI Table I row remains results-only |

The current repository has 51 checked-in BenchmarkQC point systems and 27
corrected MolVQE-21 point systems. The 51 BenchmarkQC entries include the
original legacy archives, accepted Fe₂S₂ historical/control variants, and the
15 reconstructed SI Table I rows.

## Reconstruction method

The 15 rows that were not already represented by a usable repository payload
were reconstructed with `tools/reconstruct_si_table_i.py` from the read-only
QCANT `change3_benchmark10_reference` source snapshot. The tool imports only
portable active-space integrals and orbital/provenance data, regenerates the
Jordan--Wigner Hamiltonian, recomputes the target-spin CASCI energy, and
promotes the case only when it agrees with both the SI machine-readable value
and the QCANT reference within `1e-8` Hartree.

Fe₂S₂ uses the exact CAS(10e,10o) integral/projector records for the published
and bridge-stretched geometries. U₂ uses the validated singlet certificate
CI vectors and CASSCF-refined active-space frames at 2.4300 and 2.8000 Å;
this preserves the corrected active-space state rather than substituting the
older active-HF partial archive.

## U₂-specific result

There are new U₂ cases beyond the current legacy CAS(6e,6o) pair:

- `U2response/overleaf_supplementary_n2_gcim/figures/u2_table_s2/results/equilibrium_r2p4300/table_s2_hamiltonian.npz`
- `U2response/overleaf_supplementary_n2_gcim/figures/u2_table_s2/results/stretched_r2p8000/table_s2_hamiltonian.npz`

The local file-provider metadata reports both SI-side files as `dataless`.
They were not copied. Instead, the validated QCANT singlet certificates were
used to reconstruct the two release payloads, and the metadata records that
provenance. The older `Benchmark-QC/old/standalone_molecules/U2/` tree also
contains a fully downloaded CAS(8e,8o) point at 2.4000 Å and a dataless
CAS(10e,10o) listing; these remain historical working artifacts rather than
SI Table I source payloads.

## What was not imported

The supporting-data archives contain the SI LaTeX table, active-space
classification, and unified validation/result records. The `Benchmark-QC`
artifact store also contains generated calculation outputs and older workflow
products. Those application artifacts were not copied; only validated input
integrals, Hamiltonians, and the minimum provenance needed to reproduce the
dataset records were retained.

No application calculations, results, checkpoints, plots, or logs from either
OneDrive tree were copied into this repository.
