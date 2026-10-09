import argparse
import csv
from pathlib import Path

from engine import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", choices=("aligned", "independent"), default=("aligned", "independent"))
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ("condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples",
            "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value",
            "secondary_metric", "secondary_value", "status_note")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for condition in args.conditions:
            for seed in args.seeds:
                for r in run(seed, condition, args.updates, args.lr):
                    writer.writerow({"condition": condition, "world_or_seed": seed, "method": r.method,
                        "serialized_bytes": r.payload_bytes, "train_tokens_or_examples": r.examples,
                        "optimizer_updates": r.updates, "active_compute_proxy": r.active_macs_proxy,
                        "wall_time_s": f"{r.wall_time_s:.6f}", "primary_metric": "test_mse",
                        "primary_value": f"{r.test_mse:.10g}", "secondary_metric": "max_layer_test_mse",
                        "secondary_value": f"{r.layer_mse_max:.10g}",
                        "status_note": f"throughput_examples_s={r.throughput_examples_s:.4f}"})
                    print(condition, seed, r.method, f"mse={r.test_mse:.6g}",
                          f"bytes={r.payload_bytes}", f"wall={r.wall_time_s:.3f}s", flush=True)


if __name__ == "__main__":
    main()
