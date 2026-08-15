# Contributing

For molecular data, follow `docs/ADDING_DATASETS.md` and run the full test suite
before opening a pull request:

```sh
python -m pip install -r requirements-validation.txt
python -m pip install -e . --no-deps
pytest -q
```

Keep scientific-data changes focused. Preserve historical archives, document model
changes as new variants, include checksums and provenance, and never commit
credentials or machine-specific paths.
