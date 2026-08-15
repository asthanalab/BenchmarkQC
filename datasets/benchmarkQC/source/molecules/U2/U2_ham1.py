"""Backwards-compatible wrapper for notebooks.

Historical notebooks import `build_u2_reference` and `H_gen` from
`datasets/benchmarkQC/source/molecules/U2/U2_ham1.py`.
The implementation now lives in the `benchmark_qc` package.

Docs:
- docs/USAGE.md
- docs/API.md
"""

from __future__ import annotations

import sys
from pathlib import Path


# Make the src-layout package importable from a notebook launched in this folder.
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from benchmark_qc.u2 import H_gen, build_u2_reference  # noqa: E402

__all__ = ["H_gen", "build_u2_reference"]
