import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import METHODS  # noqa: E402
from run import evaluate, load_model, make_data, payload_arrays  # noqa: E402


def verify(result_path, payload_dir):
    result = json.loads(Path(result_path).read_text())
    data, mix = make_data(result["seed"])
    assert np.allclose(mix.numpy(), result["target_mix_matrix"], rtol=0, atol=1e-8)
    checked = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{result['seed']}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        model = load_model(method, payload_arrays(path), result["seed"])
        for split in ("validation", "test"):
            actual = evaluate(model, data[split])
            expected = row[split]
            for key in ("mean_source_mse", "mean_target_mse", "macs_per_target_example"):
                assert np.isclose(actual[key], expected[key], rtol=0, atol=1e-12), (method, split, key)
            for key in ("source_mses", "target_mses", "fusion_coefficients", "source_matrix_norms"):
                assert np.allclose(actual[key], expected[key], rtol=0, atol=1e-12), (method, split, key)
        checked.append({"method": method, "bytes": len(raw), "sha256": digest, "replay": "exact"})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
