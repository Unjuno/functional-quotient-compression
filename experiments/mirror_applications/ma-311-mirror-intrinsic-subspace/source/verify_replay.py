from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

FIELDS = ("serialized_bytes", "parameter_tensor_bytes", "support_examples", "test_examples",
          "optimizer_updates", "fit_compute_proxy", "active_ops_per_example", "active_ops_total",
          "mse", "normalized_mse")


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replay_file(path, seeds, lane):
    rows = read_rows(path)
    assert {int(r["world_or_seed"]) for r in rows} == set(seeds)
    assert len(rows) == len(seeds) * 2 * 5
    mismatches, max_diff = [], 0.0
    for seed in seeds:
        for kind in ("aligned", "unrelated"):
            actual_rows = run_world(seed, kind, EXP / "artifacts" / f"replay_{lane}")
            expected = {(r["condition"], r["method"], int(r["world_or_seed"])): r for r in rows}
            for actual in actual_rows:
                old = expected[(kind, actual["method"], seed)]
                for key in FIELDS:
                    a, b = float(actual[key]), float(old[key])
                    diff = abs(a - b)
                    max_diff = max(max_diff, diff)
                    if diff > (1e-12 if key in ("mse", "normalized_mse") else 0):
                        mismatches.append({"lane": lane, "seed": seed, "kind": kind,
                                           "method": actual["method"], "field": key, "difference": diff})
                if actual["payload_sha256"] != old["payload_sha256"]:
                    mismatches.append({"lane": lane, "seed": seed, "kind": kind,
                                       "method": actual["method"], "field": "payload_sha256"})
    return rows, max_diff, mismatches


def main():
    dev, dev_diff, dev_bad = replay_file(EXP / "RESULTS_CORE.csv", [31101, 31102, 31103], "dev")
    fresh_path = EXP / "FRESH_RESULTS.csv"
    if fresh_path.exists():
        fresh, fresh_diff, fresh_bad = replay_file(fresh_path, [31111, 31112, 31113], "fresh")
    else:
        fresh, fresh_diff, fresh_bad = [], 0.0, []
    result = {"rows_replayed": len(dev) + len(fresh), "max_absolute_difference": max(dev_diff, fresh_diff),
              "payload_hashes_exact": not (dev_bad or fresh_bad), "mismatches": dev_bad + fresh_bad,
              "fresh_opened": bool(fresh), "development_fresh_split_integrity": "passed"}
    (HERE / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if dev_bad or fresh_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
