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


def replay(path, seeds, lane):
    rows = read_rows(path)
    assert {int(r["world_or_seed"]) for r in rows} == set(seeds)
    assert len(rows) == len(seeds) * 10
    diffs, bad = [], []
    for seed in seeds:
        for kind in ("aligned", "unrelated"):
            generated = run_world(seed, kind, EXP / "artifacts" / f"replay_{lane}")
            table = {(r["condition"], int(r["world_or_seed"]), r["method"]): r for r in rows}
            for actual in generated:
                old = table[(kind, seed, actual["method"])]
                for field in FIELDS:
                    d = abs(float(actual[field]) - float(old[field]))
                    diffs.append(d)
                    if d > (1e-12 if field in ("mse", "normalized_mse") else 0):
                        bad.append({"lane": lane, "seed": seed, "kind": kind, "method": actual["method"], "field": field})
                if actual["payload_sha256"] != old["payload_sha256"]:
                    bad.append({"lane": lane, "seed": seed, "kind": kind, "method": actual["method"], "field": "payload_sha256"})
    return len(rows), max(diffs, default=0), bad


def main():
    n, d, bad = replay(EXP / "RESULTS_CORE.csv", [31201, 31202, 31203], "dev")
    fresh_path = EXP / "FRESH_RESULTS.csv"
    fresh_opened = fresh_path.exists()
    if fresh_opened:
        n2, d2, bad2 = replay(fresh_path, [31211, 31212, 31213], "fresh")
    else:
        n2, d2, bad2 = 0, 0, []
    result = {"rows_replayed": n + n2, "max_absolute_difference": max(d, d2),
              "payload_hashes_exact": not (bad or bad2), "mismatches": bad + bad2,
              "fresh_opened": fresh_opened, "development_fresh_split_integrity": "passed"}
    (HERE / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if bad or bad2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
