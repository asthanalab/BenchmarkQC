# Dataset catalog

`datasets/catalog.json` indexes the 90
checked-in point-level Hamiltonian archives: 63 BenchmarkQC systems and 27
corrected MolVQE-21 systems. SHA-256 checksums allow each binary file to be
verified before use. Each geometry and active-space/orbital variant is a
separate catalog row.

| System and model | Geometry coverage | Status | Data |
| --- | --- | --- | --- |
| C2 aug-cc-pVTZ CAS(8e,8o), state-specific CASCI natural orbitals | 2.2000 Angstrom | Current | `datasets/benchmarkQC/source/molecules/C2/cas8e8o_augccpvtz/casci_natural_orbitals/C2_PES_H.npz` |
| N2 STO-6G CAS(4e,4o) | 11 points, 0.8-2.8 Angstrom | Historical, preserved unchanged | `datasets/benchmarkQC/source/molecules/N2/N2_PES_H1.npz` |
| N2 cc-pVDZ CAS(6e,6o), ASF-selected space | 12 points, 0.8-3.0 Angstrom | Reconstructed CASCI series; original Figure 1 numerical archive unavailable | `datasets/benchmarkQC/source/molecules/N2/cas6e6o_ccpvdz/N2_PES_H.npz` |
| N2 cc-pVDZ CAS(10e,8o), canonical RHF | 1.0977 and 2.0000 Angstrom | Current control | `datasets/benchmarkQC/source/molecules/N2/cas10e8o_ccpvdz/canonical/N2_PES_H.npz` |
| N2 cc-pVDZ CAS(10e,8o), CASCI natural orbitals | 1.0977 and 2.0000 Angstrom | Current final calculation | `datasets/benchmarkQC/source/molecules/N2/cas10e8o_ccpvdz/casci_natural_orbitals/N2_PES_H.npz` |
| FeS ANO-RCC-MB CAS(6e,6o), quintet | 14 points | Existing | `datasets/benchmarkQC/source/molecules/FeS/FeS_PES_H.npz` |
| Fe2S2 Li--Chan-parent CAS(4e,4o), singlet | 1 archived point; coordinates unavailable | Current; accepted historical frame | `datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/accepted/historical_cas4e4o/Fe2S2_H.npz` |
| Fe2S2 Li--Chan-parent CAS(6e,6o), singlet | 1 archived point; coordinates unavailable | Current; accepted historical frame | `datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/accepted/historical_cas6e6o/Fe2S2_H.npz` |
| Fe2S2 Li--Chan-parent CAS(8e,6o), singlet | 1 archived point; coordinates unavailable | Current; accepted historical frame | `datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/accepted/historical_cas8e6o/Fe2S2_H.npz` |
| Fe2S2 parent-frame CAS(8e,8o), singlet | 1 archived point; coordinates unavailable | Current; accepted constructed nested control | `datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/accepted/nested_cas8e8o/Fe2S2_H.npz` |
| U2 SF-X2C-1e/ANO-RCC-MB CAS(6e,6o), singlet | 2.40 and 2.48 Angstrom | Existing partial grid | `datasets/benchmarkQC/source/molecules/U2/U2_PES_H.npz` |
| SI Table I reconstructed rows | FeS, Fe₂S₂, U₂, C₂H₄, CH₂, C₂, O₂, and FeH; one catalog row per geometry | Validated reconstruction | `datasets/benchmarkQC/*/systems/` |
| MolVQE-21 corrected CASSCF/natural-orbital reference cases | 27 one-point case archives | Current corrected-cache import | `datasets/molvqe21/systems/` |

The full SI Table I inventory is at
[`si_table_i_inventory.json`](../datasets/benchmarkQC/source/si_table_i_inventory.json).
It identifies 18 active-space models spanning nine molecular species and 32
representative geometry values. A complete PES can contain additional catalog
points: the new N2 CAS(6e,6o) model contributes twelve, not twelve SI models.

Every catalog row includes a normalized numeric integral archive in addition to
the historical PennyLane-object representation. Where the original orbital frame
was not retained, the archive is explicitly marked as a JW-equivalent
reconstruction rather than presented as original AO/MO data. Scalar CASCI,
CISD, CCSD, Rényi, and cumulant records and availability statuses for the 63 BenchmarkQC rows are in
[`reference_results.json`](../datasets/benchmarkQC/reference/reference_results.json) and
are copied into each point's `metadata.json`.

The catalog describes only files actually present in this repository. It does
not create entries for systems whose source Hamiltonians have not been added.

For N2 CAS(6e,6o), the CASCI reference agrees with the visible Figure 1 curve
within image-reading precision. The original numerical archive was not recovered,
and agreement of the GCIM error curve is not claimed. Saved densities, orbital
coefficients, both orbital frames, and AO-level regeneration tests accompany
the Hamiltonians. See the model's `README.txt` for exact reproduction commands.

## MolVQE-21 corrected cache set

The `datasets/molvqe21/` folder is sourced from `asthanaa/benchmarkkrylov` commit
`2f1faef5589dd276393be4cb93860524f4ff790a`. It includes the 27 cases present
in that commit’s corrected-cache manifest, together with normalized numeric
integrals, checksums, source-cache provenance, and the explicit active-orbital
override records used for cyclobutadiene and m-benzyne. The two source records
without corrected benchmark caches are intentionally excluded.

## Fe2S2 accepted records and ASF candidate

The source and acceptance workflow for Li--Chan-parent Fe2S2 CAS(4e,4o),
CAS(6e,6o), and CAS(8e,6o) historical benchmarks, plus a modern ASF
CAS(6e,6o) comparison, is present at `datasets/benchmarkQC/source/molecules/Fe2S2/cas6e6o_chan30e20o/`. The three
historical variants passed promotion and are the three Fe2S2 rows above. Each
has a checksum-pinned Hamiltonian, pickle-free source-integral archive, and
acceptance report.

The fourth accepted Fe2S2 row is a present-work nested CAS(8e,8o) control in
the same qualified parent frame.  It activates parent MOs 11--18 and strictly
contains the historical CAS(6e,6o) span.  Its PySCF 2.4 primary calculation
and PySCF 2.11 cross-check, exact-state diagnostics, and cross-version
certificate are retained under `accepted/nested_cas8e8o/`.  It is not a
historical paper active space.

The accepted records reproduce the paper's historical default-RHF/CASCI partitions.
The shared RHF calculation stopped before orbital stationarity, so these models
must not be described as conventional converged-RHF, natural-orbital, or
Active Space Finder selections. The FCIDUMP does not contain coordinates or AO
basis labels; consequently, the stored label `0.0` is a point identifier rather
than a bond length. The modern DMRG/Active Space Finder CAS(6e,6o) route remains
a separate, unaccepted candidate and is not a catalog row.
