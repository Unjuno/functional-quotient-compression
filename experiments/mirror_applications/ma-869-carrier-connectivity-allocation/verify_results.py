"""Deterministic replay of MA-869 fresh metrics (timings are excluded)."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from experiment import run

saved = json.loads((ROOT / "fresh_results.json").read_text())
replayed = run([701, 719, 733])
for key in ("central", "best_ignoring_connectivity", "connectivity_aware", "random_reachable", "storage"):
    for metric, expected in saved[key].items():
        actual = replayed[key][metric]
        if isinstance(expected, float):
            assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-14), (key, metric, expected, actual)
        else:
            assert actual == expected, (key, metric, expected, actual)
for row_saved, row_actual in zip(saved["rows"], replayed["rows"]):
    assert row_saved["world"] == row_actual["world"]
    for policy in ("central", "best_ignoring_connectivity", "connectivity_aware", "random_reachable"):
        for metric, expected in row_saved[policy].items():
            assert row_actual[policy][metric] == expected, (policy, metric)
print("MA-869 fresh metric replay: exact; world assignments: exact")
