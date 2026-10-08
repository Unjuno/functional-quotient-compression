from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

FRESH = [26611, 26612, 26613]
FIELDS = ["condition", "split", "world_or_seed", "method", "serialized_bytes",
          "parameter_tensor_bytes", "examples", "optimizer_updates", "active_ops_per_example",
          "active_ops_total", "encode_wall_time_s", "wall_time_s", "examples_per_s", "mse",
          "normalized_mse", "payload_sha256", "oracle_factor_ids"]


def main() -> None:
    with (EXP / "RESULTS_CORE.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    aligned = [r for r in rows if r["condition"] == "aligned" and r["split"] == "heldout"]
    mirror = [r for r in aligned if r["method"] == "mirror"]
    simple = [r for r in aligned if r["method"] in ("tied", "vera_sum", "coeff_product")]
    simple_at_gate = [r for r in simple if float(r["normalized_mse"]) <= 1e-5]
    indep_bytes = [int(r["serialized_bytes"]) for r in aligned if r["method"] == "independent"]
    gate = (len(mirror) == 3
            and max(float(r["normalized_mse"]) for r in mirror) <= 1e-5
            and sum(int(r["serialized_bytes"]) for r in mirror) / len(mirror) <= 0.5 * sum(indep_bytes) / len(indep_bytes)
            and bool(simple_at_gate)
            and sum(int(r["serialized_bytes"]) for r in mirror) / len(mirror)
            <= 0.9 * min(int(r["serialized_bytes"]) for r in simple_at_gate))
    if not gate:
        raise RuntimeError("Frozen full development Mirror-specific gate failed; fresh seeds remain sealed")
    out = EXP / "artifacts" / "fresh"
    fresh_rows = []
    for seed in FRESH:
        for kind in ("aligned", "unrelated"):
            fresh_rows.extend(run_world(seed, kind, out))
    with (EXP / "FRESH_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fresh_rows)
    print(json.dumps({"gate": "passed", "fresh_rows": len(fresh_rows)}, indent=2))


if __name__ == "__main__":
    main()
