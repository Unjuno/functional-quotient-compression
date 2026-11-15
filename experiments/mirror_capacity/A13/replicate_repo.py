# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Portable A13c runner: use frozen replicate.py as-is, with sibling A12 imports."""
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent
A12=HERE.parent/'A12'
if A12.is_dir():
    sys.path.insert(0,str(A12))
    import a12  # noqa: F401 - loaded under same module name before frozen runner
# In the evidence ZIP, the originally frozen source is bundled instead.
runpy.run_path(str(HERE/'replicate.py'),run_name='__main__')
