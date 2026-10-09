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
    data = make_data(seed)
    checked = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{seed}_{method}.npz"
        raw = path.read_bytes()
        assert len(raw) == row["serialized_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"]
        model = load_model(method, payload_arrays(path), seed)
        for split in ("validation", "test"):
            actual = evaluate(model, method, data[split])
            expected = row[split]
            for key in ("macro_nll", "macro_accuracy", "held_out_macro_nll", "held_out_macro_accuracy", "seen_macro_nll"):
                assert np.isclose(actual[key], expected[key], rtol=0, atol=1e-12), (method, split, key)
            for got, want in zip(actual["configurations"], expected["configurations"]):
                for key in ("nll", "accuracy", "macs_per_example"):
                    assert np.isclose(got[key], want[key], rtol=0, atol=1e-12), (method, split, got, want)
        checked.append({"method": method, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "replay": "exact"})
    return {"seed": seed, "condition": result["condition"], "payloads": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
