#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p artifacts results logs
python -m pytest -q tests/test_core.py followup_tests.py
python -u source/run.py --stage calibration
python -u source/run.py --stage fresh
python -u verify.py
python - <<'PY'
import json
from pathlib import Path
import followup
Path('results/FOLLOWUP_LOCK.json').write_text(json.dumps({'hashes':followup.hashes(),'new_worlds_not_opened':True},indent=2))
PY
python -u followup.py
python analyze.py
