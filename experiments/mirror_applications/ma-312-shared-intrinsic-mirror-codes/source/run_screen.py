from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

SEEDS = [31201, 31202, 31203]
FIELDS = ["condition", "world_or_seed", "method", "serialized_bytes", "parameter_tensor_bytes",
          "support_examples", "test_examples", "optimizer_updates", "fit_compute_proxy",
          "active_ops_per_example", "active_ops_total", "encode_wall_time_s", "wall_time_s",
          "examples_per_s", "mse", "normalized_mse", "payload_sha256"]


def main():
    rows = []
    for seed in SEEDS:
        for kind in ("aligned", "unrelated"):
            rows.extend(run_world(seed, kind, EXP / "artifacts" / "development"))
    with (EXP / "RESULTS_CORE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"development_rows": len(rows), "fresh_opened": False}, indent=2))


if __name__ == "__main__":
    main()
