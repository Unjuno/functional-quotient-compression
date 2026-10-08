import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

import run

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "artifacts" / "fresh"
checked = 0
max_abs_diff = 0.0
for summary_path in sorted(FRESH.glob("*.json")):
    record = json.loads(summary_path.read_text())
    seed = record["seed"]
    data = run.world(seed)
    target = run.reference_outputs(data[3], data[4], data[5])
    for row in record["summaries"]:
        payload = FRESH / f"fresh_{seed}_{row['method']}.zip"
        raw = payload.read_bytes()
        assert len(raw) == row["serialized_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"]
        arrays = run.load_arrays(payload)
        metrics = run.evaluate(row["method"], arrays, target, data[5])
        delta = abs(metrics["output_nmse"] - row["output_nmse"])
        max_abs_diff = max(max_abs_diff, delta)
        assert delta <= 1e-15
        assert np.allclose(metrics["layer_output_nmse"], row["layer_output_nmse"], rtol=0, atol=1e-15)
        checked += 1
print(json.dumps({"payloads_and_metric_rows_checked": checked,
                  "max_abs_output_nmse_difference": max_abs_diff}))
