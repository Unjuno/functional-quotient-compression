from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

FRESH = [31111, 31112, 31113]
FIELDS = ["condition", "world_or_seed", "method", "serialized_bytes", "parameter_tensor_bytes",
          "support_examples", "test_examples", "optimizer_updates", "fit_compute_proxy",
          "active_ops_per_example", "active_ops_total", "encode_wall_time_s", "wall_time_s",
          "examples_per_s", "mse", "normalized_mse", "payload_sha256"]


def main():
    with (EXP / "RESULTS_CORE.csv").open(encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    aligned = [r for r in dev if r["condition"] == "aligned"]
    methods = {m: [r for r in aligned if r["method"] == m] for m in ("mirror", "coeff2", "said4")}
    per_world = []
    for seed in (31101, 31102, 31103):
        m = next(r for r in methods["mirror"] if int(r["world_or_seed"]) == seed)
        c = next(r for r in methods["coeff2"] if int(r["world_or_seed"]) == seed)
        s = next(r for r in methods["said4"] if int(r["world_or_seed"]) == seed)
        per_world.append(float(m["normalized_mse"]) <= 1e-4
                         and abs(float(m["normalized_mse"]) - float(c["normalized_mse"])) <= 1e-5
                         and int(m["serialized_bytes"]) <= 0.9 * int(c["serialized_bytes"])
                         and int(m["serialized_bytes"]) <= 0.75 * int(s["serialized_bytes"]))
    if len(aligned) != 15 or not all(per_world):
        raise RuntimeError("Frozen full development gate failed; fresh seeds remain sealed")
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
