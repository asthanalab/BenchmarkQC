# Integral archives

Every one-point entry in [`datasets/catalog.json`](../datasets/catalog.json)
has a normalized, pickle-free spatial active-space archive. The archive stores
the same fields for BenchmarkQC and corrected MolVQE-21 cases:

- `geometry_angstrom`
- `one_body_integrals` in spatial-orbital form
- `two_body_integrals` in PySCF chemist ordering, with
  ( (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq) )
- `core_constant`
- active-space determinant and orbital-selection metadata

Use `benchmark_qc.integral_dataset.load_spatial_integral_archive` to validate
an archive and `jordan_wigner_terms_from_integrals` to regenerate its qubit
Hamiltonian. Spin orbitals are interleaved: even wires are alpha and odd wires
are beta.

## Historical JW-equivalent archives

The original historical N2, FeS, and U2 scans retained a PennyLane JW operator but
not the numeric active-space integral/orbital frame. For those 27 point-level
records, `tools/recover_missing_integrals.py` solves the inverse JW linear map
over real spatial one-/two-body coefficients and writes a normalized archive.
The resulting tensor is checked by regenerating the saved operator; the maximum
observed Pauli-coefficient error is below `4e-11`.

These archives are valid numeric representatives of the published Hamiltonian.
They should not be interpreted as the original AO/MO coefficients, and
orbital-frame-sensitive classical quantities should use the scalar reference
records and source provenance instead. The distinction is recorded in each
system's `metadata.json` under `integrals.frame_status` and
`provenance.integral_reconstruction`.

To repeat the recovery after restoring the source catalog and dependencies:

```sh
PYTHONPATH=src python tools/recover_missing_integrals.py
```

The tool updates only catalog entries without an integral path. It performs no
application calculations; those belong under the ignored `applications/`
workspace.
