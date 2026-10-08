import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import METHODS, IndependentWidths, SharedSlimmableMLP  # noqa: E402
from run import evaluate, make_data, payload_arrays  # noqa: E402


def verify(result_path, payload_dir):
    result = json.loads(Path(result_path).read_text())
    seed = result["seed"]
    splits = make_data(seed)
    reports = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{seed}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        arrays = payload_arrays(path)
        if method == "independent":
            model = IndependentWidths(seed * 37 + 5)
        else:
            model = SharedSlimmableMLP(method, seed * 37 + METHODS.index(method))
        ref = model.state_dict()
        import torch
        state = {}
        for key, value in ref.items():
            tensor = torch.from_numpy(np.array(arrays[key], copy=True))
            state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
        model.load_state_dict(state)
        model.eval()
        val = evaluate(model, method, splits["validation"])
        test = evaluate(model, method, splits["test"])
        for key, actual, expected in (
            ("validation_width_nll", val["width_nll"], row["validation_width_nll"]),
            ("validation_width_accuracy", val["width_accuracy"], row["validation_width_accuracy"]),
            ("test_width_nll", test["width_nll"], row["test_width_nll"]),
            ("test_width_accuracy", test["width_accuracy"], row["test_width_accuracy"]),
        ):
            assert np.allclose(actual, expected, rtol=0, atol=1e-12), (method, key, actual, expected)
        reports.append({"method": method, "bytes": len(raw), "sha256": digest, "replay": "exact"})
    return {"seed": seed, "condition": result["condition"], "payloads": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
