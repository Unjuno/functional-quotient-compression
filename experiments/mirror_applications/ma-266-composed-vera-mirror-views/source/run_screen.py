from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from engine import run_world  # noqa: E402

SEEDS = [26601, 26602, 26603]
FIELDS = ["condition", "split", "world_or_seed", "method", "serialized_bytes",
          "parameter_tensor_bytes", "examples", "optimizer_updates", "active_ops_per_example",
          "active_ops_total", "encode_wall_time_s", "wall_time_s", "examples_per_s", "mse",
          "normalized_mse", "payload_sha256", "oracle_factor_ids"]


def main() -> None:
    out = EXP / "artifacts" / "development"
    rows = []
    for seed in SEEDS:
        for kind in ("aligned", "unrelated"):
            rows.extend(run_world(seed, kind, out))
    with (EXP / "RESULTS_CORE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "seeds": SEEDS, "fresh_opened": False}, indent=2))


if __name__ == "__main__":
    main()
