"""Check prompt payload bytes, retrieval, and selected-prompt function replay."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import PROMPTS, PromptBank, bank_from_payload, nearest_key, select_prompt_outputs  # noqa: E402
from run import make_world, nrmse  # noqa: E402


def verify(result_path: Path, payload_dir: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    world = make_world(result["seed"])
    checked = []
    common_keys = None
    true_ids = torch.arange(PROMPTS).repeat_interleave(result["test_examples_per_prompt"])
    for row in result["results"]:
        path = payload_dir / row["payload_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            payload = {k: archive[k] for k in archive.files}
        bank = bank_from_payload(payload)
        keys = torch.as_tensor(payload["retrieval_keys"], dtype=torch.float32)
        if common_keys is None:
            common_keys = keys
        else:
            assert torch.equal(common_keys, keys), "all methods must use identical keys"
        query = world["query_test"]
        prompt_ids = nearest_key(query, keys)
        accuracy = float((prompt_ids == true_ids).float().mean())
        with torch.no_grad():
            all_output = bank(world["x_test"])
            oracle = select_prompt_outputs(all_output, true_ids)
            retrieved = bank.forward_selected(world["x_test"], prompt_ids)
            oracle_error = nrmse(oracle, world["target_test"])
            retrieved_by_prompt = [
                nrmse(retrieved.reshape(PROMPTS, result["test_examples_per_prompt"], -1)[i],
                      world["target_test_task"][i])
                for i in range(PROMPTS)
            ]
            retrieved_error = float(np.mean(retrieved_by_prompt))
        assert np.isclose(accuracy, row["retrieval_top1_accuracy"], rtol=0, atol=1e-12)
        assert np.isclose(oracle_error, row["source_test_nrmse"], rtol=0, atol=1e-7)
        assert np.isclose(retrieved_error, row["retrieved_test_nrmse"], rtol=0, atol=1e-7)
        checked.append({"method": row["method"], "bytes": len(raw), "sha256": digest,
                        "retrieval_top1_accuracy": accuracy,
                        "oracle_test_nrmse": oracle_error, "retrieved_test_nrmse": retrieved_error,
                        "retrieved_test_nrmse_by_prompt": retrieved_by_prompt})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": checked,
            "same_keys_and_metric_replay": "PASS"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("payload_dir", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
