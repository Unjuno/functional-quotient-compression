from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

FRESH = [31211, 31212, 31213]
FIELDS = ["condition", "world_or_seed", "method", "serialized_bytes", "parameter_tensor_bytes",
          "support_examples", "test_examples", "optimizer_updates", "fit_compute_proxy",
          "active_ops_per_example", "active_ops_total", "encode_wall_time_s", "wall_time_s",
          "examples_per_s", "mse", "normalized_mse", "payload_sha256"]


def main():
    with (EXP / "RESULTS_CORE.csv").open(encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    aligned = [r for r in dev if r["condition"] == "aligned"]
    ok = len(aligned) == 15
    for seed in (31201, 31202, 31203):
        get = lambda method: next(r for r in aligned if r["method"] == method and int(r["world_or_seed"]) == seed)
        mirror, coeff, said = get("mirror"), get("coeff2"), get("said8")
        ok = (ok and float(mirror["normalized_mse"]) <= 1e-4
              and abs(float(mirror["normalized_mse"]) - float(coeff["normalized_mse"])) <= 1e-5
              and int(mirror["serialized_bytes"]) <= 0.9 * int(coeff["serialized_bytes"])
              and int(mirror["serialized_bytes"]) <= 0.75 * int(said["serialized_bytes"]))
    if not ok:
        raise RuntimeError("Frozen development gate failed; fresh task streams remain sealed")
    rows = []
    for seed in FRESH:
        for kind in ("aligned", "unrelated"):
            rows.extend(run_world(seed, kind, EXP / "artifacts" / "fresh"))
    with (EXP / "FRESH_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"development_gate": "passed", "fresh_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
