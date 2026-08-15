# N2 datasets

This folder retains the original N2 benchmark and adds the current
CAS(10e,8o)/cc-pVDZ revision dataset without overwriting historical files.

| Dataset | Geometries | Status |
| --- | ---: | --- |
| `N2_PES_H1.npz` | 11 points, 0.8-2.8 Angstrom | Historical STO-6G CAS(4e,4o); unchanged |
| `cas10e8o_ccpvdz/canonical/N2_PES_H.npz` | 1.0977 and 2.0000 Angstrom | Canonical-RHF control |
| `cas10e8o_ccpvdz/casci_natural_orbitals/N2_PES_H.npz` | 1.0977 and 2.0000 Angstrom | Final CASCI-natural-orbital calculation |

See `cas10e8o_ccpvdz/README.md` and `cas10e8o_ccpvdz/metadata.json` for the
complete model definition, source archives, reproducibility commands, validation
results, and checksums.
