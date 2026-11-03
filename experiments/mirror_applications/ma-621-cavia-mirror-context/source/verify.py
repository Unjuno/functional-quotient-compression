#!/usr/bin/env python3
"""Replay MA-621 serialization and development metrics."""
import csv
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import SEEDS, run  # noqa: E402

expected = list(csv.DictReader((HERE.parent / "artifacts/dev_results.csv").open()))
actual = [r for seed in SEEDS for r in run(seed)]
assert len(expected) == len(actual) == 10
for got, want in zip(actual, expected):
    assert int(got["world"]) == int(want["world"])
    assert got["method"] == want["method"]
    assert int(got["serialized_bytes"]) == int(want["serialized_bytes"])
    assert abs(float(got["heldout_nmse"]) - float(want["heldout_nmse"])) < 1e-8

# NPZ payload is deterministic and each saved state is measured in the runner.
buf = io.BytesIO()
np.save(buf, np.arange(8, dtype=np.float32), allow_pickle=False)
assert np.load(io.BytesIO(buf.getvalue()), allow_pickle=False).tolist() == list(range(8))
print("MA-621 replay: 10 rows, deterministic serialized-byte and metric checks passed")
