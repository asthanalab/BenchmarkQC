# N2 CAS(10e,8o)/cc-pVDZ

This directory contains the two-point N2 dataset produced for the manuscript
revision: the equilibrium geometry (1.0977 Angstrom) and one stretched geometry
(2.0000 Angstrom). Both datasets use a neutral singlet CAS(10e,8o) model in the
cc-pVDZ basis and map 16 interleaved spin orbitals to qubits with the
Jordan-Wigner transformation.

## Included variants

| Variant | Orbital basis | Hamiltonian archive |
| --- | --- | --- |
| Canonical control | Geometry-specific canonical RHF orbitals | `canonical/N2_PES_H.npz` |
| Final natural-orbital calculation | Geometry-specific spin-summed exact-CASCI natural orbitals | `casci_natural_orbitals/N2_PES_H.npz` |

Each Hamiltonian archive follows the repository's existing three-array format:
`labels`, `Hs`, and `casci_energies`. The numeric source archives under each
variant's `inputs/` directory additionally preserve the geometry, one- and
two-electron integrals, core constant, orbital coefficients, active-orbital
indices, AO overlap, and reference determinant. The natural-orbital sources also
preserve the natural rotation, occupations, active coefficients, and CASCI
one-particle density matrices.

The natural orbitals use exact target-state information and rotate only within
the unchanged eight-orbital active subspace. They are not CASSCF-optimized
orbitals. Interpret this variant as an orbital-preconditioning sensitivity test,
with the canonical dataset as the unpreconditioned control.

## Reproduce and validate

From the repository root:

```sh
python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/generate_hamiltonians.py --write --output-root /tmp/benchmarkqc-n2
python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/generate_hamiltonians.py --check
python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/test_hamiltonians.py --variant casci_natural_orbitals --index 0
python datasets/benchmarkQC/N2/cas10e8o_ccpvdz/test_hamiltonians.py --variant casci_natural_orbitals --index 1
```

The exact energy check is performed in the physical 10-electron singlet sector,
not in unrestricted Fock space. Scientific definitions, reference values,
software versions, and SHA-256 checksums are recorded in `metadata.json`.
The generator never overwrites catalogued archives unless `--write --force` is
explicitly requested; its normal/default mode is the non-mutating integrity check.

## Security note

`N2_PES_H.npz` contains pickled PennyLane objects for backward compatibility.
Use `allow_pickle=True` only for files obtained from a trusted source. The input
archives are numeric-only and load with `allow_pickle=False`.
