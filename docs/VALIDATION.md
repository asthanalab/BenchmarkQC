# Validation and release checks

The repository distinguishes three levels of validation:

1. **Schema checks** verify that every Hamiltonian archive contains exactly
   `labels`, `Hs`, and `casci_energies`, with the catalog point count and
   checksum.
2. **Physics checks** compare saved reference energies in the appropriate
   fixed-electron and fixed-spin sector. The tests include the high-spin FeS
   sector and the corrected MolVQE-21 cases.
3. **Reproducibility checks** rebuild Jordan–Wigner terms from the numeric
   integral archives and compare them coefficient-by-coefficient with the
   checked-in Hamiltonians.

Run the complete local gate from the repository root:

```sh
python -m pip install -e .
python -m pytest -q
python -m build
```

The CI workflow runs the test suite on Python 3.11 and 3.12 and performs the
syntax/undefined-name lint gate. Dataset-specific builders support `--check`
without writing into the checked-in archive; use a directory under
`applications/scratch/` or a system temporary directory for generated output.

Before publishing a change, also review:

```sh
git diff --check
git status --short
git check-ignore -v applications/results/example.json
```

Do not add application outputs, credentials, hardware job identifiers, or
machine-specific absolute paths to the dataset commit.
