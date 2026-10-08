from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import METHODS, make_world, make_state, load_payload, run_world, save_payload  # noqa: E402

COMPARE = ("serialized_bytes", "parameter_tensor_bytes", "examples", "optimizer_updates",
           "active_ops_per_example", "active_ops_total", "mse", "normalized_mse")


def rows_from(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    dev = rows_from(EXP / "RESULTS_CORE.csv")
    expected_seeds = {26601, 26602, 26603}
    assert {int(r["world_or_seed"]) for r in dev} == expected_seeds
    assert not (EXP / "FRESH_RESULTS.csv").exists(), "fresh data should remain sealed after a development gate failure"
    assert len(dev) == 60
    mismatches = []
    max_diff = 0.0
    for seed in sorted(expected_seeds):
        for condition in ("aligned", "unrelated"):
            replay = run_world(seed, condition, EXP / "artifacts" / "replay")
            expected = {(r["method"], r["split"]): r for r in dev
                        if int(r["world_or_seed"]) == seed and r["condition"] == condition}
            for row in replay:
                old = expected[(row["method"], row["split"])]
                for field in COMPARE:
                    a, b = float(row[field]), float(old[field])
                    diff = abs(a - b)
                    max_diff = max(max_diff, diff)
                    tol = 1e-12 if field in ("mse", "normalized_mse") else 0
                    if diff > tol:
                        mismatches.append({"seed": seed, "condition": condition,
                                           "method": row["method"], "split": row["split"],
                                           "field": field, "difference": diff})
                if row["payload_sha256"] != old["payload_sha256"]:
                    mismatches.append({"seed": seed, "condition": condition,
                                       "method": row["method"], "field": "payload_sha256"})
    result = {"checked_rows": len(dev), "max_absolute_difference": max_diff,
              "payload_hashes_exact": not mismatches, "mismatches": mismatches,
              "development_fresh_split_integrity": "passed; fresh unopened"}
    (HERE / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
