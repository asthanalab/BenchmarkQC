# Fe2S2 benchmark data

The current Fe2S2 work is under `cas6e6o_chan30e20o/`. It starts from the
checksum-pinned CAS(30e,20o) FCIDUMP distributed with the Li--Chan
spin-projected MPS study and provides explicitly separated reduction
workflows:

- faithful sibling reconstructions of the paper's historical CAS(4e,4o),
  CAS(6e,6o), and CAS(8e,6o) default-RHF/CASCI partitions; and
- an accepted present-work CAS(8e,8o) control that strictly contains the
  historical CAS(6e,6o) span in the same qualified parent frame; and
- a new strict fixed-size Active Space Finder candidate selected from DMRG
  pair information in the full CAS(30e,20o) parent.

No candidate is exposed through `datasets/catalog.json` until its desktop or
scheduler-backed report passes every applicable exact-sector, energy, RDM,
provenance, and checksum gate.  The constructed CAS(8e,8o) record additionally
retains an independent PySCF 2.4/2.11 cross-version certificate.
