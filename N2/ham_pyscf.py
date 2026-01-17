"""Backwards-compatible wrapper for notebooks.

Historical notebooks import `H_gen` from `N2/ham_pyscf.py`.
The implementation now lives in the `benchmark_qc` package.

Docs:
- docs/USAGE.md
- docs/API.md
"""

from __future__ import annotations

import sys
from pathlib import Path


# Make repo root importable even when running from inside the N2 folder.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_qc.n2 import H_gen  # noqa: E402

__all__ = ["H_gen"]
