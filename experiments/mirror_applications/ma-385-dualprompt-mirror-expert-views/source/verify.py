import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from run import METHODS, evaluate, load_serialized_model, make_data


def read_payload(path):
    with zipfile.ZipFile(path) as archive:
        return {name[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(name)), allow_pickle=False)
                for name in archive.namelist()}


def same_metrics(actual, expected, method, label):
    for metric in ("accuracy", "bce", "retrieval_accuracy"):
        assert np.isclose(actual[metric], expected[metric], rtol=0, atol=1e-12), (method, label, metric)
    assert np.allclose(actual["per_task_accuracy"], expected["per_task_accuracy"], rtol=0, atol=1e-12)


def verify(result_path, outdir):
    result = json.loads(Path(result_path).read_text())
    data = make_data(result["seed"])
    reports = []
    for row in result["summaries"]:
        method, seed = row["method"], result["seed"]
        final_path = Path(outdir) / "payloads" / f"{result['condition']}_{seed}_{method}.npz"
        raw = final_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["serialized_bytes"] and digest == row["payload_sha256"]
        arrays = read_payload(final_path)
        assert int(arrays["meta"][3]) == METHODS.index(method)
        model, keys = load_serialized_model(method, seed, arrays)
        same_metrics(evaluate(model, data["validation"], keys), row["validation"], method, "validation")
        same_metrics(evaluate(model, data["test"], keys), row["test"], method, "test")
        snapshots = []
        for stage in row["sequential_validation_history"]:
            task_count = stage["tasks_seen"]
            snap_path = Path(outdir) / "snapshots" / f"{result['condition']}_{seed}_{method}_seen_{task_count}.npz"
            snap_raw = snap_path.read_bytes()
            snap_digest = hashlib.sha256(snap_raw).hexdigest()
            assert len(snap_raw) == stage["diagnostic_snapshot_bytes"]
            assert snap_digest == stage["diagnostic_snapshot_sha256"]
            snap_arrays = read_payload(snap_path)
            snap_model, snap_keys = load_serialized_model(method, seed, snap_arrays)
            got = evaluate(snap_model, data["validation"], snap_keys, max_task=task_count - 1)
            same_metrics(got, stage["validation"], method, f"tasks_seen_{task_count}")
            snapshots.append({"tasks_seen": task_count, "bytes": len(snap_raw), "sha256": snap_digest})
        reports.append({"method": method, "final_bytes": len(raw), "sha256": digest,
                        "final_metric_replay": "exact", "retention_snapshots": snapshots})
    return {"experiment_id": "MA-385", "seed": result["seed"], "condition": result["condition"],
            "methods": reports, "fresh_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("outdir")
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.outdir), indent=2))
