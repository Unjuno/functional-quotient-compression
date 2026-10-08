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

DEV_SEEDS = [25801, 25802, 25803]


def main() -> None:
    out = EXP / "artifacts" / "development_payloads"
    # Frozen development-only rank selection; no fresh rows are accessed.
    rank_rows = []
    for rank in (1, 2, 4, 8):
        for seed in DEV_SEEDS:
            teachers, _, _, angles = __import__("engine").make_world(seed, "aligned")
            from engine import save_payload, load_payload, evaluate, svd_codec
            state = svd_codec(teachers, rank)
            path = out / f"rank_{rank}_{seed}.npz"
            byte_count = save_payload(path, state)
            state2 = load_payload(path)
            _, x, y, _ = __import__("engine").make_world(seed, "aligned")
            m = evaluate("svd", state2, x, y, teachers, rank=rank)
            rank_rows.append({"rank": rank, "seed": seed, "bytes": byte_count, **m})
    (EXP / "source" / "dev_rank_sweep.json").write_text(json.dumps(rank_rows, indent=2) + "\n")
    independent_bytes = []
    for seed in DEV_SEEDS:
        teachers, _, _, _ = __import__("engine").make_world(seed, "aligned")
        from engine import save_payload
        independent_bytes.append(save_payload(out / f"ind_ref_{seed}.npz", {"weights": teachers}))
    byte_gate = 0.5 * float(np.mean(independent_bytes))
    eligible = [r for r in rank_rows if r["bytes"] <= byte_gate]
    if not eligible:
        raise RuntimeError("No SVD rank qualifies under the preregistered byte gate")
    chosen_rank = min({r["rank"] for r in eligible},
                      key=lambda k: (np.mean([r["normalized_mse"] for r in eligible if r["rank"] == k]), k))
    (EXP / "source" / "dev_selection.json").write_text(json.dumps({
        "selected_svd_rank": chosen_rank,
        "selection_rule": "lowest development aligned mean normalized MSE among ranks whose actual serialized bytes <=50% independent; lower rank breaks exact tie",
        "independent_mean_bytes": float(np.mean(independent_bytes)),
        "byte_gate": byte_gate,
        "eligible_ranks": sorted({r["rank"] for r in eligible}),
        "rank_means": {str(k): float(np.mean([r["normalized_mse"] for r in eligible if r["rank"] == k]))
                       for k in sorted({r["rank"] for r in eligible})}
    }, indent=2) + "\n")

    rows = []
    for seed in DEV_SEEDS:
        for kind in ("aligned", "unrelated"):
            rows.extend(run_world(seed, kind, rank=chosen_rank, output_dir=out))
    fields = ["condition", "world_or_seed", "method", "serialized_bytes", "parameter_tensor_bytes",
              "examples", "optimizer_updates", "active_macs_per_example", "active_macs_total",
              "encode_wall_time_s", "wall_time_s", "examples_per_s", "mse", "normalized_mse", "status_note"]
    results_path = EXP / "RESULTS_CORE.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "rank_sweep_rows": len(rank_rows), "selected_rank": chosen_rank,
                      "results": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
