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
    seed = result["seed"]
    data, world = make_data(seed)
    assert world == result["world_coefficients"]
    reports = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{seed}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        model = load_model(method, payload_arrays(path))
        for split in ("validation", "test"):
            actual = evaluate(model, data[split])
            expected = row[split]
            for key in ("final_nll", "final_accuracy", "mean_depth_nll", "mean_depth_accuracy"):
                assert np.isclose(actual[key], expected[key], rtol=0, atol=1e-12), (method, split, key)
            for got, want in zip(actual["depth_prefixes"], expected["depth_prefixes"]):
                for key in ("nll", "accuracy", "macs_per_token"):
                    assert np.isclose(got[key], want[key], rtol=0, atol=1e-12), (method, split, got, want)
        reports.append({"method": method, "bytes": len(raw), "sha256": digest, "replay": "exact"})
    return {"seed": seed, "condition": result["condition"], "world_coefficients": world, "payloads": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
