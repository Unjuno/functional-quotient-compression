"""Replay reported quality/byte/compute metrics from the frozen seeds."""
import csv
import json
import math
import sys
from pathlib import Path

from engine import run_one


def main(path):
    rows = list(csv.DictReader(open(path, newline="")))
    checks = 0
    max_delta = 0.0
    for row in rows:
        got = run_one(row["method"], int(row["seed"]), row["condition"])
        for source_key, result_key in (("serialized_bytes", "payload_bytes"), ("decode_macs", "decode_macs"),
                                       ("examples", "examples"), ("optimizer_updates", "optimizer_updates"),
                                       ("tokens", "tokens")):
            assert int(row[source_key]) == int(got[result_key]), (row, source_key, got[result_key])
        for key in ("activation_mse", "relative_fro"):
            delta = abs(float(row[key]) - float(got[key]))
            max_delta = max(max_delta, delta)
            assert math.isclose(float(row[key]), float(got[key]), rel_tol=1e-7, abs_tol=1e-10), (row, key, got[key])
        checks += 1
    report = {"experiment_id": "MA-160", "rows": checks, "payload_and_compute_exact": True,
              "quality_replay_max_abs_difference": max_delta, "fresh_seeds": [16011, 16012, 16013],
              "development_seeds": [16001, 16002], "note": "Timing is measured on replay and is not required to match the original wall-clock sample."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main(Path(sys.argv[1]))
