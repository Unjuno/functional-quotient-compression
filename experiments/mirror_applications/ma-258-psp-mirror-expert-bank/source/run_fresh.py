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

FRESH_SEEDS = [25811, 25812, 25813]


def main() -> None:
    # Check the frozen development gate before any fresh-world construction.
    with (EXP / "RESULTS_CORE.csv").open(encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    aligned = [r for r in dev if r["condition"] == "aligned"]
    mirror = [r for r in aligned if r["method"] == "mirror"]
    independent = float(np.mean([int(r["serialized_bytes"]) for r in aligned if r["method"] == "independent"]))
    simple = [r for r in aligned if r["method"] in ("svd", "psp", "tied")]
    best_simple = min(float(r["normalized_mse"]) for r in simple)
    mirror_bytes = float(np.mean([int(r["serialized_bytes"]) for r in mirror]))
    simple_at_quality = [r for r in simple if float(r["normalized_mse"]) <= 1e-6]
    gate = (len(mirror) == 3
            and max(float(r["normalized_mse"]) for r in mirror) <= 1e-6
            and mirror_bytes <= 0.5 * independent
            and bool(simple_at_quality)
            and mirror_bytes <= 0.9 * min(int(r["serialized_bytes"]) for r in simple_at_quality))
    if not gate:
        raise RuntimeError("Frozen development gate failed; fresh worlds remain sealed")
    selection = json.loads((HERE / "dev_selection.json").read_text())
    rank = int(selection["selected_svd_rank"])
    out = EXP / "artifacts" / "fresh_payloads"
    rows = []
    for seed in FRESH_SEEDS:
        for kind in ("aligned", "unrelated"):
            rows.extend(run_world(seed, kind, rank=rank, output_dir=out))
    fields = ["condition", "world_or_seed", "method", "serialized_bytes", "parameter_tensor_bytes",
              "examples", "optimizer_updates", "active_macs_per_example", "active_macs_total",
              "encode_wall_time_s", "wall_time_s", "examples_per_s", "mse", "normalized_mse", "status_note"]
    with (EXP / "FRESH_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"development_gate": "passed", "rank": rank, "fresh_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
