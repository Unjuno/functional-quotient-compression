"""Replay MA-431 locked-result aggregates and check frozen split/payload records."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "RESULTS_CORE.csv"
rows = list(csv.DictReader(RESULTS.open()))
assert len(rows) == 45, f"expected 45 fresh rows, found {len(rows)}"
assert {int(r["world"]) for r in rows} == {43110, 43111, 43112}
assert {int(r["seed"]) for r in rows} == {0, 1, 2}
assert {r["method"] for r in rows} == {"static", "rank1", "universal_lowrank", "mirror", "independent"}
assert all(float(r["learning_rate"]) == 0.001 for r in rows)
assert all(float(r["coordinate_scale"]) == 1.0 for r in rows)
assert all(int(r["updates"]) == 1500 for r in rows)
assert all(len(r["sha256"]) == 64 and int(r["payload_bytes"]) > 0 for r in rows)

summary = {}
for method in sorted({r["method"] for r in rows}):
    subset = [r for r in rows if r["method"] == method]
    summary[method] = {
        key: statistics.mean(float(r[key]) for r in subset)
        for key in ("relative_mse", "payload_bytes", "active_macs_proxy", "train_seconds", "inference_seconds", "examples_per_second")
    }
for world in (43110, 43111, 43112):
    w = [r for r in rows if int(r["world"]) == world]
    mirror = statistics.mean(float(r["relative_mse"]) for r in w if r["method"] == "mirror")
    independent = statistics.mean(float(r["relative_mse"]) for r in w if r["method"] == "independent")
    assert mirror <= 1.05 * independent, f"world {world} missed per-world quality gate: {mirror} vs {independent}"
    mb = statistics.mean(float(r["payload_bytes"]) for r in w if r["method"] == "mirror")
    ib = statistics.mean(float(r["payload_bytes"]) for r in w if r["method"] == "independent")
    assert mb <= 0.80 * ib, f"world {world} missed byte gate: {mb} vs {ib}"
print(json.dumps({"rows": len(rows), "fresh_world_quality_and_byte_gates": "PASS", "summary": summary}, indent=2, sort_keys=True))
