N2 CAS(6e,6o)/cc-pVDZ: main Figure 1 CASCI reconstruction
======================================================

One active-space model (SI case 18), twelve bond lengths in Angstrom:
0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.8, 3.0.
The SI table lists representative values at 1.2 and 2.0 Angstrom.

Each systems/n2_ccpvdz_cas6e6o_fixed_size_asf_r* folder uses the same layout
and loaders as every other BenchmarkQC system:
  hamiltonian.npz              labels, Hs, casci_energies
  metadata.json                definitions, references and qualifications
  inputs/source_integrals.npz  numeric integrals, geometry and full orbital input

N2_PES_H.npz here contains all twelve points in the same archive format.
The Hamiltonians use Jordan-Wigner mapping with alpha on even spin-orbital
indices and beta on odd indices. Energies include nuclear repulsion and the
four inactive doubly occupied orbitals.

Construction follows an internally stable molecular UHF solution along the
geometry grid. ASF 2.0.1 uses spin-summed DF-UMP2 natural-orbital candidates,
occupation bounds 0.005/1.995, an initial twelve-orbital window, and fixed-size
six-electron/six-orbital selection. Source natural orbitals 4--9 are selected
at every geometry. At 0.8 Angstrom the unfiltered fixed-size fallback is used.
The space has N 2p sigma/sigma-star and pi/pi-star character.

The exported frame is a canonical stationary active-only RHF rotation of this
selected span. Both frames, their rotation, the original density, full orbital
coefficients, explicit basis definition, occupations and geometry are retained.
CASCI is unchanged by this active-space rotation. Determinant entropy, CISD,
CCSD and an adaptive trajectory need not be unchanged.

Regenerate using the saved orbitals, not a new unconstrained SCF/ASF search:
  PYTHONPATH=src python -m benchmark_qc.n2_cas6e6o --repository .
  PYTHONPATH=src python -m pytest tests/test_n2_cas6e6o.py

The validator rebuilds AO integrals, compares every Pauli coefficient and
solves the 400-dimensional fixed-number, M_S=0 sector. Independent export
checks compare the entire sector matrix against the PySCF determinant matrix.
All twelve CASCI roots are singlets. CISD is independently diagonalized in
the 55-dimensional singlet sector of the 118-determinant CISD space, rather
than choosing a potentially higher-spin lowest M_S=0 root at stretched bonds.
CCSD is provided for the two SI points;
other points are explicitly not evaluated, not described as failed.

Agreement with Figure 1 is limited by its image resolution: the maximum
difference from the digitized CASCI curve is 0.349455 mEh, within the
conservative two-pixel reading uncertainty of 1.171155 mEh. The original
numerical archive was not recovered. These are reproducible reconstructed
Hamiltonians, not proof of identity with the original coefficients. The
GCIM error curve has not been reproduced. No energy shift or scale is fitted.

Full-precision references and selection checks:
  selection_and_reference_records.json
  hamiltonian_validation.json
  cisd_validation.json
  figure1_comparison.json
