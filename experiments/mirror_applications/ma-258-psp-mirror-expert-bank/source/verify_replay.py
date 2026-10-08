from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

FIELDS = ("serialized_bytes", "parameter_tensor_bytes", "examples", "optimizer_updates",
          "active_macs_per_example", "active_macs_total", "mse", "normalized_mse")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    dev_seeds = {25801, 25802, 25803}
    fresh_seeds = {25811, 25812, 25813}
    dev = load_rows(EXP / "RESULTS_CORE.csv")
    fresh = load_rows(EXP / "FRESH_RESULTS.csv")
    assert {int(r["world_or_seed"]) for r in dev} == dev_seeds
    assert {int(r["world_or_seed"]) for r in fresh} == fresh_seeds
    assert dev_seeds.isdisjoint(fresh_seeds)
    assert len(dev) == len(fresh) == 30
    rank = int(json.loads((HERE / "dev_selection.json").read_text())["selected_svd_rank"])
    mismatches = []
    max_abs = 0.0
    for lane, rows in (("dev", dev), ("fresh", fresh)):
        for seed in sorted({int(r["world_or_seed"]) for r in rows}):
            for kind in ("aligned", "unrelated"):
                replay = run_world(seed, kind, rank, EXP / "artifacts" / f"replay_{lane}")
                saved = {(r["method"]): r for r in rows
                         if int(r["world_or_seed"]) == seed and r["condition"] == kind}
                for row in replay:
                    old = saved[row["method"]]
                    for field in FIELDS:
                        actual = float(row[field])
                        expected = float(old[field])
                        diff = abs(actual - expected)
                        max_abs = max(max_abs, diff)
                        tol = 1e-12 if field in ("mse", "normalized_mse") else 0
                        if diff > tol:
                            mismatches.append({"lane": lane, "seed": seed, "condition": kind,
                                               "method": row["method"], "field": field,
                                               "expected": expected, "actual": actual, "difference": diff})
                    if row["status_note"].split("payload_sha256=")[1].split(";")[0] != \
                            old["status_note"].split("payload_sha256=")[1].split(";")[0]:
                        mismatches.append({"lane": lane, "seed": seed, "condition": kind,
                                           "method": row["method"], "field": "payload_sha256"})
    result = {"checked_rows": len(dev) + len(fresh), "max_absolute_difference": max_abs,
              "mismatches": mismatches, "split_integrity": "passed"}
    (HERE / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
