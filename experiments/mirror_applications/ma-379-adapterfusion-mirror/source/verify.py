"""Check on-disk payload bytes/hashes and replay source/fusion test metrics."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import TARGETS, SourceBank, bank_from_payload  # noqa: E402
from run import make_world, nrmse  # noqa: E402


def verify(result_path: Path, payload_dir: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    world = make_world(result["seed"])
    checked = []
    for row in result["results"]:
        path = payload_dir / row["payload_path"]
        raw = path.read_bytes()
        assert len(raw) == row["serialized_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            payload = {k: archive[k] for k in archive.files}
        bank = bank_from_payload(payload).eval()
        router = torch.as_tensor(payload["router"], dtype=torch.float32)
        with torch.no_grad():
            source = bank(world["x_test"])
            source_error = nrmse(source, world["source_test"])
            weights = torch.softmax(torch.einsum("bd,tdn->btn", world["x_test"], router), dim=-1)
            fused = torch.einsum("btn,bnd->btd", weights, source)
            fusion_errors = [nrmse(fused[:, t], world["target_test"][:, t]) for t in range(TARGETS)]
        assert np.isclose(source_error, row["source_test_nrmse"], rtol=0, atol=1e-7)
        assert np.allclose(fusion_errors, row["fusion_test_nrmse_by_target"], rtol=0, atol=1e-7)
        checked.append({"method": row["method"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "source_test_nrmse": source_error, "fusion_test_nrmse_by_target": fusion_errors})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": checked,
            "byte_hash_and_metric_replay": "PASS"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("payload_dir", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
