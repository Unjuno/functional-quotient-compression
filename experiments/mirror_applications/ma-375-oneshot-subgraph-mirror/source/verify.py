import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import METHODS  # noqa: E402
from run import evaluate, load_model, make_data, payload_arrays, spearman  # noqa: E402


def verify(result_path, payload_dir):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    independent_test = next(r["test"]["paths"] for r in result["summaries"] if r["method"] == "independent")
    target = [r["nll"] for r in independent_test]
    reports = []
    for row in result["summaries"]:
        method = row["method"]
        path = Path(payload_dir) / f"{result['condition']}_{result['seed']}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"]
        assert digest == row["payload_sha256"]
        model = load_model(method, payload_arrays(path), result["seed"])
        for split in ("validation", "test"):
            actual = evaluate(model, method, data[split])
            expected = row[split]
            for got, want in zip(actual["paths"], expected["paths"]):
                for key in ("nll", "accuracy", "macs_per_example"):
                    assert np.isclose(got[key], want[key], rtol=0, atol=1e-12), (method, split, key)
        if method != "independent":
            rank = spearman([r["nll"] for r in row["validation"]["paths"]], target)
            assert np.isclose(rank, row["validation_to_independent_test_spearman"], rtol=0, atol=1e-12)
        reports.append({"method": method, "bytes": len(raw), "sha256": digest, "replay": "exact"})
    return {"seed": result["seed"], "condition": result["condition"], "payloads": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("payload_dir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.payload_dir), indent=2))
