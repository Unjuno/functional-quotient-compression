import hashlib
import json
from pathlib import Path

import numpy as np

import run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "development_A3"
checked = 0
max_diff = 0.0
gate_rows = []
for metrics_path in sorted(RESULTS.glob("32*.json")):
    record = json.loads(metrics_path.read_text())
    seed = record["seed"]
    _, transitions = run.teacher_world(seed)
    splits = run.sample_pairs(seed, transitions)
    for row in record["summaries"]:
        payload = RESULTS / "payloads" / f"development_{seed}_{row['method']}.npz"
        raw = payload.read_bytes()
        assert len(raw) == row["serialized_bytes"], (payload, len(raw), row["serialized_bytes"])
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"], payload
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        model = run.load_payload_model(row["method"], arrays, seed * 100 + run.METHODS.index(row["method"]))
        for split_name, metric_key in (("validation", "validation_domain_nll"), ("test", "domain_nll")):
            evaluation = run.evaluate(model, splits[split_name])
            replay = evaluation["domain_nll"]
            if split_name == "validation":
                row["replayed_validation_domain_accuracy"] = evaluation["domain_accuracy"]
            recorded = row[metric_key]
            diff = max(abs(a - b) for a, b in zip(replay, recorded))
            max_diff = max(max_diff, diff)
            assert np.allclose(replay, recorded, rtol=0, atol=1e-12), (seed, row["method"], split_name, replay, recorded)
        checked += 1
        gate_rows.append({"seed": seed, "method": row["method"], "validation_domain_nll": row["validation_domain_nll"], "validation_domain_accuracy": row.get("replayed_validation_domain_accuracy"), "serialized_bytes": row["serialized_bytes"]})
summary = {"payloads_checked": checked, "metric_replay_max_abs_difference": max_diff, "fresh_opened": False, "validation_gate_rows": gate_rows}
(RESULTS / "replay.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"payloads_checked": checked, "metric_replay_max_abs_difference": max_diff, "fresh_opened": False}, indent=2))
