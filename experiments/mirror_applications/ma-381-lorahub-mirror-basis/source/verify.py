"""Serialized replay and LoRA functional gauge verification for MA-381."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CANDIDATES, INPUT_DIM, TARGETS, bank_from_payload  # noqa: E402
from run import gauge_audit, make_world, nrmse  # noqa: E402


def verify(result_path: Path, payload_dir: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    world = make_world(result["seed"])
    checked = []
    for row in result["results"]:
        path = payload_dir / row["payload_path"]
        raw = path.read_bytes()
        assert len(raw) == row["serialized_bytes"]
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == row["payload_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            payload = {k: archive[k] for k in archive.files}
        bank = bank_from_payload(payload)
        coeff = torch.as_tensor(payload["composition"], dtype=torch.float32)
        with torch.no_grad():
            source = bank(world["x_source_test"])
            source_error = nrmse(source, world["source_test"])
            flat = world["x_target_test"].reshape(-1, INPUT_DIM)
            candidate = bank(flat).reshape(TARGETS, 2048, CANDIDATES, 16)
            prediction = torch.einsum("tn,tbnd->tbd", coeff, candidate)
            target_by_task = [nrmse(prediction[t], world["target_test"][t]) for t in range(TARGETS)]
        assert np.isclose(source_error, row["source_test_nrmse"], rtol=0, atol=1e-7)
        assert np.allclose(target_by_task, row["target_test_nrmse_by_task"], rtol=0, atol=1e-7)
        gauge = gauge_audit(bank, result["seed"])
        if row["method"] == "independent":
            assert gauge is not None
            assert max(gauge.values()) <= 1e-6, gauge
            assert max(gauge.values()) <= max(row["gauge_audit"].values()) + 1e-7
        else:
            assert gauge is None
        checked.append({"method": row["method"], "bytes": len(raw), "sha256": digest,
                        "source_test_nrmse": source_error, "target_test_nrmse_by_task": target_by_task,
                        "gauge_audit": gauge})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": checked,
            "serialized_metric_replay": "PASS"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("payload_dir", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
