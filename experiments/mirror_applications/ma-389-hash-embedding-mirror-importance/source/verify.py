"""Replay serialized MA-389 payloads and check their recorded measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import HashEmbeddingBank, bank_from_payload
from run import make_world, score


def verify(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    world = make_world(int(result["seed"]))
    checks = []
    for row in result["results"]:
        path = result_path.parent / row["payload_path"]
        raw = path.read_bytes()
        assert len(raw) == row["serialized_bytes"], (path, "byte count")
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"], (path, "sha256")
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        bank = bank_from_payload(payload)
        assert bank.method == row["method"]
        replayed = score(bank, world)
        for metric, value in replayed.items():
            assert abs(value - row[metric]) <= 1e-7, (path, metric, value, row[metric])
        checks.append({"method": row["method"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "replayed_metrics": replayed})
    return {"experiment_id": "MA-389", "seed": result["seed"], "condition": result["condition"],
            "payloads_checked": len(checks), "all_exact": True, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.result)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
