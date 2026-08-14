# Local application workspace

This directory is intentionally not a dataset or a package. It is a safe,
local workspace for running algorithms against the checked-in reference
archives.

The following subdirectories are ignored by Git:

- `runs/` — raw optimizer, sampler, or hardware-run outputs;
- `results/` — derived numerical results and tables;
- `plots/` — generated figures;
- `logs/` — application logs;
- `scratch/` — disposable working files.

Keep only reusable application code in a separate project or in a reviewed
source directory. Do not add application calculations, credentials, hardware
job identifiers, or generated result files to the Benchmark-QC dataset commit.
