import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

import run

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "artifacts" / "fresh"
checked = 0
max_abs_diff = 0.0
for summary_path in sorted(FRESH.glob("*.json")):
    record = json.loads(summary_path.read_text())
    seed = record["seed"]
    basis, _, targets = run.world(seed)
    pairs = run.split_pairs(seed, targets)
    for row in record["summaries"]:
        payload = FRESH / f"fresh_{seed}_{row['method']}.zip"
        raw = payload.read_bytes()
        assert len(raw) == row["serialized_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"]
        arrays = {}
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for member in z.namelist():
                name = member[:-4]
                arrays[name] = np.load(io.BytesIO(z.read(member)), allow_pickle=False)
        assert np.array_equal(arrays["meta"], np.asarray([4, 4, 16, 16, 2, 2], dtype=np.uint16))
        method_idx = run.METHODS.index(row["method"])
        method_seed = seed if row["method"] in ("ordinary_product", "mirror_product") else seed + method_idx * 13
        model = run.load_model(row["method"], arrays, method_seed, basis)
        test = run.metric(model, pairs["test"])
        delta = abs(test["mean_nmse"] - row["test_nmse"])
        max_abs_diff = max(max_abs_diff, delta)
        assert delta <= 1e-15
        assert np.allclose(test["pair_mse"], row["test_pair_nmse"], rtol=0, atol=1e-15)
        expected_held = float(np.mean([test["pair_mse"][i*run.EXPERTS+i] for i in range(run.LAYERS)])) if row["method"] in ("ordinary_product", "mirror_product") else None
        assert expected_held == row["heldout_test_nmse"]
        checked += 1
print(json.dumps({"payloads_and_metric_rows_checked": checked, "max_abs_test_nmse_difference": max_abs_diff}))
