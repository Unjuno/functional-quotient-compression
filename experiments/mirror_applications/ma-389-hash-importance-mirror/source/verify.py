import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from run import METHODS, evaluate, load_serialized, make_data


def read_payload(path):
    with zipfile.ZipFile(path) as archive:
        return {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                for name in archive.namelist()}


def verify(result_path, outdir):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    reports = []
    for row in result["summaries"]:
        method, seed = row["method"], result["seed"]
        path = Path(outdir) / "payloads" / f"{result['condition']}_{seed}_{method}.npz"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"] and digest == row["payload_sha256"]
        arrays = read_payload(path)
        assert int(arrays["meta"][4]) == METHODS.index(method)
        if method != "independent":
            assert np.array_equal(arrays["hash_params"], data["hash_params"])
        model, indices = load_serialized(method, seed, arrays)
        for split in ("validation", "test"):
            actual = evaluate(model, data[split], indices, data)
            expected = row[split]
            for metric in ("accuracy", "nll", "collided_token_accuracy", "unique_signature_accuracy", "collision_rate"):
                assert np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-12), (method, split, metric)
        reports.append({"method": method, "bytes": len(raw), "sha256": digest, "metric_replay": "exact"})
    return {"experiment_id": "MA-389", "seed": result["seed"],
            "unique_hash_signatures": result["unique_hash_signatures"],
            "collided_token_fraction": result["collided_token_fraction"],
            "methods": reports, "fresh_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("outdir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.outdir), indent=2))
