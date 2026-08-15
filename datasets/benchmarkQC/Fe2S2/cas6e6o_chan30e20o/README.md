# Fe2S2 reduced spaces from the Chan CAS(30e,20o) Hamiltonian

This folder contains the actual Li--Chan parent Hamiltonian and all readable
source needed to derive, validate, and publish the paper's 8- and 12-qubit
Fe2S2 benchmarks. The
parent is `parent/chan_fe2s2_cas30e20o.FCIDUMP` (SHA-256
`95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9`),
with `NORB=20`, `NELEC=30`, and `MS2=0`.

The FCIDUMP is sufficient to reproduce the numerical Hamiltonian, but it does
not encode coordinates, AO basis labels, MO coefficients, or chemical orbital
labels. Those fields remain explicitly unavailable rather than being guessed.

## Which construction produced the paper energy?

The recovered source is preserved byte-for-byte at
`provenance/vi66cycle_historical_source.py` (SHA-256
`388897d39451399f506b8d293c6e2465db9882deec61aa5da1e70075dec612b7`).
It shows that the historical CAS(6e,6o) Hamiltonian was not selected by Active
Space Finder. It ran PySCF's default RHF on the Chan integrals and then used a
default CASCI orbital partition. The same replayed RHF frame and default
partition rule define the paper's CAS(4e,4o), CAS(6e,6o), and CAS(8e,6o)
sibling models, with respective paper targets `-116.154749`, `-116.156259`, and
`-116.158572` Ha. These are independent reductions and are not labeled as a
nested convergence sequence. The RHF calculation stopped at the historical
RHF cycle limit, so the accepted record must disclose the nonstationary parent
frame. Replaying it gives `-116.15625982795386` Ha with PySCF 2.4; the reported
`-116.156259` Ha is the value truncated to six decimal places. The independent
PySCF 2.11 cross-check agrees at the approximately 10-picohartree level.

The validated primary PySCF 2.4 values are:

| Space | Active parent MOs (zero-based) | CASCI (Ha) | CISD (Ha) | CCSD (Ha) | H0.25 (nats) | C2 | N92 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAS(4e,4o) | 13--16 | -116.15474998779426 | -116.15474589468376 | -116.15474974192813 | 1.532814066066578 | 0.2425186811604375 | 2 |
| CAS(6e,6o) | 12--17 | -116.15625982795386 | -116.15618492267850 | -116.15619403159668 | 3.0221215489221076 | 0.28329738176500885 | 2 |
| CAS(8e,6o) | 11--16 | -116.15857228045772 | -116.15854603556220 | -116.15856619743963 | 2.856612621074474 | 0.2702597782719671 | 2 |

The shared parent RHF stopped unconverged with orbital-gradient norm
`4.845940182351517e-4`; the accepted historical records retain this fact in
both metadata and catalog claim boundaries.

## Constructed nested CAS(8e,8o) control

`generate_nested_cas8e8o.py` applies the same recovered default-partition
recipe to a larger space.  It freezes parent-frame MOs 0--10, activates MOs
11--18, and leaves MO 19 external.  The resulting CAS(8e,8o) span therefore
strictly contains the historical CAS(6e,6o) span (MOs 12--17).  This is a
current-project constructed control, not one of the three active spaces
reported in the paper and not an Active Space Finder selection.

The accepted primary PySCF 2.4 calculation gives CASCI
`-116.16151636450809` Ha, CISD `-116.16112026246387` Ha, and CCSD
`-116.16118628087456` Ha.  The exact singlet has `H0.25 =
4.889532973316796` nats, `C2 = 0.2606372888864521`, leading-determinant
probability `0.8718223421452893`, and `N92 = 2`.  Relative to the nested
CAS(6e,6o) model, CASCI is lower by `5.256536554242075` mHa and H0.25 is
larger, whereas C2 is smaller, N92 is unchanged, and the leading determinant
has greater weight.  The larger order-0.25 Renyi entropy therefore records a
broader low-weight coefficient tail; it is not, by itself, evidence for a
monotonic increase in multireference complexity.

An independent PySCF 2.11 replay agrees with the primary CASCI energy within
`3.1861e-11` Ha and passes the cross-version energy, descriptor, active-space
projector, integral, and pickle-free numeric-archive gates.  The accepted
record is in `accepted/nested_cas8e8o/`, including both raw calculation
reports and `validation/cross_version_certificate.json`.  It retains the
same nonstationary-parent-frame qualification as the historical records.

Run the faithful replay on the registered desktop:

```sh
python reproduce_historical_reductions.py \
  --execution-site desktop \
  --output-dir candidates/historical_reductions_pyscf2p4
```

The equivalent Talon command must be executed inside a Slurm allocation:

```sh
python reproduce_historical_reductions.py \
  --execution-site talon \
  --output-dir candidates/historical_reductions_talon
```

## New Active Space Finder construction

`generate_asf_cas6e6o.py` implements a genuinely new reduction from the same
Chan parent. It first produces converged singlet DMRG RDMs in CAS(30e,20o),
checks the parent energy against the literature value `-116.6056091` Ha, and
passes those RDMs to Active Space Finder's strict fixed-size pair-information
selector. It then solves the selected CAS(6e,6o) independently.

The paper energy is not used to select orbitals. It is applied only after ASF
has returned a space. If the independently selected ASF Hamiltonian does not
reproduce the paper value within 1 microhartree, the candidate remains a
scientifically useful comparison but is rejected as a replacement for the
historical row.

Example desktop invocation, using explicit readable paths:

```sh
python generate_asf_cas6e6o.py \
  --execution-site desktop \
  --block-executable /plain/path/to/block2main \
  --scratch-dir /plain/path/to/fe2s2_asf_scratch \
  --output-dir candidates/asf_dmrg_desktop
```

Strict selection is the default. If and only if strict ASF reports that no
CAS(6e,6o) candidate exists in its filtered pool, an explicitly authorized
unfiltered comparison can be run by adding `--fallback-unfiltered`. The report
then records `selection_mode: explicit_unfiltered_fallback`, the strict
exception, and both authorization and use of the fallback. It is never silent.

The script requires Active Space Finder, `pyscf-dmrgscf`, and a configured
Block2 executable. For Talon, use `--execution-site talon` inside Slurm and
provide the site-appropriate executable and MPI prefix. The command deliberately
has no silent unfiltered ASF fallback.

## Candidate and promotion contract

The historical command writes one subdirectory per requested space
(`cas4e4o/`, `cas6e6o/`, and `cas8e6o/`). Each historical subdirectory and the
ASF candidate directory contain three files:

- `source_integrals.npz`: pickle-free integrals, exact CI/RDM data, orbital
  provenance, correlation descriptors, and classical-control amplitudes;
- `Fe2S2_H.npz`: the backward-compatible BenchmarkQC Hamiltonian archive; and
- `candidate_report.json`: numerical results, execution provenance, acceptance
  gates, and SHA-256 checksums.

Audit without modifying the repository:

```sh
python promote_accepted_candidate.py \
  --candidate-report candidates/historical_reductions_pyscf2p4/cas4e4o/candidate_report.json
```

After review, promote the accepted candidate explicitly:

```sh
python promote_accepted_candidate.py \
  --candidate-report candidates/historical_reductions_pyscf2p4/cas4e4o/candidate_report.json \
  --promote
```

Promotion is the only workflow that writes under `accepted/` or adds Fe2S2 to
the root `datasets/catalog.json`. A rejected report, failed Boolean gate, unsupported
execution site, missing file, path traversal, or checksum mismatch stops the
promotion before catalog publication.

`metadata.template.json` and `catalog_entry.template.json` describe the final
records before any candidate is accepted. The exact H0.25 Renyi entropy,
two-body cumulant norm `C2`, N92, CISD, and CCSD values are generated from the
accepted active-space Hamiltonian and stored with its report; values from a
different Fe2S2 active space must never be substituted.
