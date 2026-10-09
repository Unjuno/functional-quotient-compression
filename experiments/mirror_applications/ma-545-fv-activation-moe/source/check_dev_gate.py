"""Check the frozen MA-545 development gate without touching fresh data."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = (54501, 54502)


def evaluate_gate(reports):
    checks = []
    for report in reports:
        result = report["results"]
        methods = result["methods"]
        oracle = methods["oracle_task_fv"]
        base = methods["no_intervention"]
        routed = methods["routed_fv_top1"]
        quality_gain = (oracle["heldout_accuracy"] - base["heldout_accuracy"] >= 0.05
                        or oracle["mean_gold_logprob"] - base["mean_gold_logprob"] >= 0.15)
        routed_accuracy_gap = abs(oracle["heldout_accuracy"] - routed["heldout_accuracy"])
        routed_logprob_gap = oracle["mean_gold_logprob"] - routed["mean_gold_logprob"]
        checks.append({"seed": report["seed"], "oracle_quality_gain": quality_gain,
                       "routed_accuracy_gap": routed_accuracy_gap,
                       "routed_gold_logprob_loss": routed_logprob_gap,
                       "pass": quality_gain and routed_accuracy_gap <= 0.05 and routed_logprob_gap <= 0.15})
    return {"seeds": [c["seed"] for c in checks], "checks": checks,
            "fresh_opened": len(checks) == 2 and all(c["pass"] for c in checks)}


def main():
    reports = [json.loads((ROOT / f"runs/dev/seed_{s}/metrics.json").read_text()) for s in DEV_SEEDS]
    gate = evaluate_gate(reports)
    (ROOT / "runs/dev/DEV_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
