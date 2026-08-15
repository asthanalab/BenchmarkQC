# Active-space and method-data audit

This document records what was checked for the 51 point-level BenchmarkQC
records. It distinguishes three different claims that are often conflated:

1. **Source fidelity:** the checked-in Hamiltonian reproduces the stated
   source calculation and reference energy.
2. **Chemical motivation:** the active orbitals span the valence or metal
   shells normally needed for the stated electronic-state problem.
3. **Method completeness:** scalar reference values for methods such as
   CASCI, CISD, CCSD, entropy, and cumulants are actually archived.

Passing the first claim does not automatically imply the second or third.
The catalog intentionally retains reduced and historical cases because they
are useful controls, but they are labeled as such below.

## Bottom line

The active-space choices are defensible for the benchmark questions for which
they were created, but they are not all universal production-quality active
spaces for every property or geometry.

- The SI-defined C₂, C₂H₄, CH₂, Fe₂S₂, FeH, FeS, N₂, O₂, and expanded U₂
  spaces have an explicit chemical definition in the source record, and the
  checked-in CASCI/Jordan--Wigner payloads pass the repository validation
  checks.
- The full-valence spaces are the strongest general choices for their target
  ground-state/bond-breaking problems: C₂ CAS(8e,8o), N₂ CAS(10e,8o), CH₂
  CAS(6e,6o), O₂ CAS(12e,8o), and the source-defined expanded FeS, FeH, and
  U₂ spaces.
- C₂H₄ CAS(2e,2o) is intentionally a π/π* torsion model, not a full-valence
  space. It is appropriate for the planar-to-90° torsion benchmark but not a
  general ethylene excited-state or dissociation space.
- The historical N₂ CAS(4e,4o), FeS CAS(6e,6o), U₂ CAS(6e,6o), and the historical
  Fe₂S₂ reductions are archive-compatible stress tests. They should not be
  presented as independently optimized, universally adequate chemical active
  spaces. The historical FeS and U₂ records also lack complete final MO
  coefficient/atom-shell labeling.
- The CASCI-natural-orbital C₂ and N₂ records rotate a fixed parent active
  subspace. They are separate benchmark variants because orbital-frame
  dependent approximate methods can change, while exact CASCI within the
  fixed subspace does not.

## Record-level assessment

Each row below applies to every geometry point in that family. A change in
bond distance is a new catalog system, but it does not by itself create a new
chemical active-space definition. For stretched points, the assessment also
states whether the selected space should be treated as a controlled model or
as a complete valence description.

| Family and catalog variants | Active-space definition | Assessment | Evidence/status |
| --- | --- | --- | --- |
| C₂, CAS(8e,8o), canonical and CASCI-natural variants | C 2s/2p valence span | Strong choice for C₂ multireference bonding and the reported low-lying valence states. The natural-orbital row is the same subspace in a target-state density frame. | SI/source definition; CASSCF(8,8)/aug-cc-pVTZ is used for C₂ potential curves in the literature. [1] |
| C₂H₄, CAS(2e,2o), planar and 90° twisted | C=C π and π* orbitals | Good minimal space for torsion and the associated π-bond breaking/diradical character. Not a full-valence space and not sufficient for arbitrary Rydberg or σ-state spectroscopy. | SI/source definition; independent torsion studies identify CAS(2,2) as sufficient for this coordinate. [2] |
| CH₂, CAS(6e,6o), 102° and 140° | C 2s/2p plus the two H 1s combinations | Good full-valence model for the stated low-lying valence problem and bent/open geometries. State-specific CASSCF solutions still require root/occupation checks because unphysical stationary solutions are known for CH₂. | SI/source definition; CH₂ literature explicitly discusses full-valence CAS(6,6) and the need to screen solutions. [3] |
| Fe₂S₂, CAS(10e,10o), two SI geometries | Ten singly occupied Fe 3d orbitals | Strong, chemically recognizable minimal space for Fe-centered spin coupling and low-energy spectra. It omits Fe double-shell and bridging-S correlation; larger CAS(10,20), CAS(22,16), or CAS(22,26) spaces are needed when those effects are part of the question. | SI/source definition; the Fe₂S₂ literature compares exactly these active-space enlargements. [4] |
| Fe₂S₂ historical CAS(4e,4o), CAS(6e,6o), CAS(8e,6o), and nested CAS(8e,8o) | Historical/default-RHF parent-frame reductions | Valid controlled orbital-space comparisons and archived reproductions. They are not claims that the reduced spaces are chemically complete. The nested CAS(8e,8o) is explicitly a present-work control, not a historical paper row. | Checksum-pinned source and acceptance reports; historical nonstationary default-RHF frame is recorded in metadata. |
| FeH, CAS(9e,10o), equilibrium and stretched | Fe 3d, 4s, 4p plus H 1s | Well motivated for the X ⁴Δ and nearby valence states in the reported diatomic problem. FeH is difficult even for high-order coupled cluster, so this should be treated as a targeted multireference benchmark rather than a universal FeH active space. | SI/source definition; FeH literature reports CASSCF calculations with CAS(9,10) and emphasizes the state sensitivity. [5] |
| FeS historical, CAS(6e,6o), 14-point scan | Six archived historical orbitals | Valid as a historical high-spin scan and reproducibility target. It is not independently auditable as a chemically complete Fe/S valence space because the archive lacks the final MO coefficient and atom/shell labels. | Source notebook and OneDrive/SI record; catalog preserves this limitation. |
| FeS expanded, CAS(14e,10o), equilibrium and stretched | Fe 3d/4s plus S 3s/3p | The chemically completed space recorded by the source is substantially better justified for the stated X ⁵Δ problem than the historical CAS(6,6) scan. It remains a targeted space and should be enlarged for ligand charge-transfer or broader spectroscopy. | SI/source definition and validated CASCI reconstruction; current benchmark preprint describes FeS as a chemically challenging active-space problem. [6] |
| N₂ historical, CAS(4e,4o), STO-6G, 11 points | N 2p π/π* selector span | Appropriate only as a deliberately reduced π/π* model. It omits the complete N 2s/2p valence span, so it should not be interpreted as the full dissociation benchmark. | Historical notebook/SI definition; no independent full-valence claim. |
| N₂, CAS(10e,8o), cc-pVDZ, canonical and CASCI-natural variants | Full N 2s/2p valence span | Strong full-valence choice for N₂ bond breaking in the stated basis. The CASCI-natural row is an orbital-frame control, not a different chemical active subspace. | SI/source definition; N₂ benchmark studies use CAS(10,8) for the cc-pVDZ dissociation problem. [7] |
| O₂, CAS(12e,8o), triplet, equilibrium and stretched | O 2s/2p valence span | Strong full-valence choice for the triplet ground-state dissociation problem. It is not intended to cover Rydberg states, charge transfer, or all spin-orbit spectroscopy. | SI/source definition; independent O₂ CASSCF work describes CAS(12,8) as the complete valence space. [8] |
| U₂ historical, CAS(6e,6o), SF-X2C-1e/ANO-RCC-MB | Fixed singlet-CASSCF orbitals optimized at 2.50 Å and projected to 2.40/2.48 Å | Useful historical fixed-orbital stress test, but too small to represent the full U₂ valence manifold. The archive also omits complete final MO coefficient/atom-shell labels. | SI/source definition; U₂ is known to be highly sensitive to active-space size and relativistic treatment. [6, 9] |
| U₂ expanded, CAS(6e,10o), SF-X2C-1e/ANO-RCC-VDZP | U 5f/6d parent span, with source ASF/CASSCF refinement | Good targeted space for the released U₂ singlet benchmark and its 2.43/2.80 Å comparison. It should not be advertised as a converged universal U₂ space: larger relativistic CASSCF spaces are used in the literature, and the source itself records strong sensitivity. | SI/source definition and validated singlet certificates; literature and the current benchmark preprint support the 5f/6d focus while warning about active-space sensitivity. [6, 9] |

## Method-data availability

The user-requested scalar-only policy is applied here: CI/CC results do not
need coefficient tensors in order to be useful as benchmark numbers. The
repository nevertheless reports exactly which scalar values are actually
present rather than implying that every method has been run for every row.

| Data item | BenchmarkQC coverage | Where it is stored |
| --- | ---: | --- |
| Jordan--Wigner Hamiltonian | 51/51 | One-point `hamiltonian.npz` archives with `labels`, `Hs`, and `casci_energies` |
| CASCI reference energy | 51/51 | Hamiltonian archive and catalog/metadata |
| Normalized one-/two-electron active-space integrals | 51/51 | `inputs/source_integrals.npz`; every archive contains numeric `one_body_integrals` and `two_body_integrals` |
| CCSD scalar energy | 49/51 | `reference_results` in each system metadata; the two expanded U₂ rows retain the SI status `validated-unavailable` |
| CISD scalar energy | 51/51 | `reference_results` in each system metadata |
| Exact-CI/CASCI coefficient data | 6/51 | Four Fe₂S₂ historical/control records and two expanded U₂ certificate records |
| RDM data | 9/51 | C₂/N₂ natural-orbital, Fe₂S₂ accepted, and expanded U₂ records |
| Rényi entropy scalar | 51/51 | `reference_results` in each system metadata; SI values are imported and missing historical points are computed from CASCI vectors |
| Two-body cumulant data/scalars | 51/51 | `reference_results` in each system metadata; full RDM tensors remain available only for the source-certified records |

The 27 historical FeS, N₂, and U₂ point archives now have the same normalized
integral and scalar-result fields as the rest of BenchmarkQC. Their metadata
clearly distinguishes values imported from SI Table I from values computed in
the recovered JW-equivalent frame. The recovered tensors reproduce the saved
qubit Hamiltonians, but they do not claim to recover unavailable original
AO/MO coefficients; orbital-frame-sensitive approximate-method values should
therefore be treated as documented controls.

## What “good” means here

For a future release, a stronger active-space certification would compare
against a systematically enlarged space at each geometry and report at least
natural occupations, orbital overlaps, state character, and the change in the
target observable. That is not the same as checking that a Hamiltonian
diagonalizes to its stored CASCI energy. The current repository therefore uses
the following release language:

- **Validated benchmark space:** source-defined and numerically reproduced;
  chemically motivated for the stated benchmark question.
- **Controlled reduced space:** deliberately smaller or orbital-frame
  variant, retained for method/orbital sensitivity studies.
- **Historical frame:** preserved for reproducibility even when the original
  archive does not expose enough orbital metadata for independent chemical
  relabeling.

The expanded SI spaces are the recommended starting point for new chemical
benchmarks. The historical records should be used when reproducing the original
algorithmic study or when explicitly studying active-space/orbital-frame
sensitivity.

## References

1. C₂ CASSCF(8,8)/aug-cc-pVTZ potential curves and low-lying states:
   [Latent harmony in dicarbon](https://pmc.ncbi.nlm.nih.gov/articles/PMC5954846/).
2. Ethylene torsion with the valence π/π* CAS(2,2) space:
   [Excited-state-specific CASSCF theory for the torsion of ethylene](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00212).
3. CH₂ full-valence CAS(6,6) and unphysical state-specific solutions:
   [Excited states, symmetry breaking, and unphysical solutions in state-specific CASSCF](https://pubs.acs.org/doi/suppl/10.1021/acs.jpca.3c00603/suppl_file/jp3c00603_si_001.pdf).
4. Fe₂S₂ active-space hierarchy:
   [Coupled cluster method tailored with quantum computing](https://doi.org/10.1021/acs.jctc.1c00589).
5. FeH low-lying states and multireference difficulty:
   [Taming the low-lying electronic states of FeH](https://pubmed.ncbi.nlm.nih.gov/23267482/).
6. Current BenchmarkQC active-space definitions and sensitivity discussion:
   [Chemically decisive benchmarks on the path to quantum utility](https://arxiv.org/abs/2601.10813).
7. N₂ cc-pVDZ CAS(10,8) dissociation benchmark:
   [Variational Quantum Eigensolver Boosted by Adiabatic Connection](https://pmc.ncbi.nlm.nih.gov/articles/PMC10823474/).
8. O₂ CAS(12,8) complete-valence CASSCF treatment:
   [CASSCF(12e,8o) O₂ dissociation curves](https://gredos.usal.es/bitstream/handle/10366/157535/PDCTQ_OrtegaAlvarezP_Cofactors.pdf%3Bjsessionid%3D755BC39F2D1E861C568EF12855A386BD?sequence=1).
9. Relativistic U₂ CASSCF and active-space sensitivity:
   [Relativistic quantum chemical calculations show that U₂ has a quadruple bond](https://www.nature.com/articles/s41557-018-0158-9).
