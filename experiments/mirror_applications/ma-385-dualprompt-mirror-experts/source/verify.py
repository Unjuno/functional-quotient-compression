"""Reload final DualPrompt payloads and replay continual-training metrics."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import INPUT_DIM, TASKS, bank_from_payload  # noqa: E402
from run import fit_bank, make_world, nrmse  # noqa: E402


def verify(result_path: Path, payload_dir: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    world = make_world(result["seed"])
    checked = []
    for row in result["results"]:
        path = payload_dir / row["payload_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            payload = {k: archive[k] for k in archive.files}
        bank = bank_from_payload(payload)
        final_errors = []
        with torch.no_grad():
            for task in range(TASKS):
                x = world["x_test"][task]
                ids = torch.full((len(x),), task, dtype=torch.long)
                pred = bank.forward_task(x, ids)
                final_errors.append(nrmse(pred, world["target_test"][task]))
        assert np.allclose(final_errors, row["final_task_nrmse"], rtol=0, atol=1e-7)
        # Replay the sequential schedule to check every old-task retention point.
        replay_bank, replay = fit_bank(row["method"], result["seed"], world)
        replay_stages = replay["stage_metrics"]
        assert [s["after_task"] for s in replay_stages] == [s["after_task"] for s in row["stage_metrics"]]
        max_stage_delta = 0.0
        for a, b in zip(replay_stages, row["stage_metrics"]):
            deltas = [abs(x-y) for x, y in zip(a["seen_task_nrmse"], b["seen_task_nrmse"])]
            max_stage_delta = max(max_stage_delta, max(deltas, default=0.0))
        assert max_stage_delta <= 1e-7
        checked.append({"method": row["method"], "bytes": len(raw), "sha256": digest,
                        "final_task_nrmse": final_errors,
                        "max_retention_stage_replay_delta": max_stage_delta})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": checked,
            "final_payload_and_sequential_retention_replay": "PASS"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("payload_dir", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
