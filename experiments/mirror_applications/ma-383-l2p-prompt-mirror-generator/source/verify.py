import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from run import METHODS, PromptModel, evaluate, make_data


def verify(result_path, payload_dir):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    reports = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{result['seed']}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == row["payload_sha256"]
        assert len(raw) == row["serialized_bytes"]
        with zipfile.ZipFile(path) as archive:
            arrays = {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                      for name in archive.namelist()}
        assert int(arrays["meta"][3]) == METHODS.index(method)
        model = PromptModel(method, result["seed"])
        state = {k: torch.tensor(arrays[k], dtype=v.dtype) for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        model.eval()
        keys = torch.tensor(arrays["router_keys"], dtype=torch.float32)
        for split in ("validation", "test"):
            actual = evaluate(model, data[split], keys)
            expected = row[split]
            for metric in ("accuracy", "bce", "retrieval_accuracy"):
                assert np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-12), (method, split, metric)
            assert np.allclose(actual["per_task_accuracy"], expected["per_task_accuracy"], rtol=0, atol=1e-12)
        reports.append({"method": method, "bytes": len(raw), "sha256": digest, "replay": "exact"})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
