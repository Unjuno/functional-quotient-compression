#!/usr/bin/env bash
# Reproduction entry for Gates 1-7 (28M lane). Requires the hash-pinned inputs
# below placed OUTSIDE git (see runs/manifest-001/official_inputs_provenance.json
# for full provenance; files live in models/ and data/, both git-ignored).
#
#   models/HF-28M/pytorch_model.bin  SHA256 8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5
#     = roneneldan/TinyStories-28M @ rev 52dabea (HF Hub) + `ln -s HF-28M models/28M`
#   data/official/TinyStories-valid.txt  SHA256 94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4
#     = roneneldan/TinyStories (HF Hub dataset)
#
#   VENV=/Users/taka/.venvs/fqc-torch  (Python 3.14.5, torch 2.14.0,
#     transformers 5.17.0, numpy 2.5.3; recorded in runs/manifest-001/)
set -euo pipefail
PY=${VENV:-/Users/taka/.venvs/fqc-torch}/bin/python
$PY scripts/fqc_preflight.py --output runs/manifest-001/preflight.json
$PY -m pytest -q tests experiments/t282/tests            # 265 passed
$PY scripts/fqc_rebuild_frozen.py --models-root ./models --output runs/rebuild-001
$PY experiments/gate1/hf_parity.py                        # needs transformers
$PY experiments/gate1/build_control.py --output runs/gate2-controls
$PY experiments/gate1/build_sharing.py --output runs/gate3-sharing
$PY experiments/gate1/build_private.py --output runs/gate4-private
$PY experiments/gate1/build_joint.py --output runs/gate35-joint
# selection evaluations (dev) then ONE audit pass after freezing candidates:
$PY experiments/gate1/batch_dev_eval.py
$PY experiments/gate1/batch_audit.py
echo REPRODUCTION_SEQUENCE_SUBMITTED
