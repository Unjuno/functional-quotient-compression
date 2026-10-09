import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma471_run", ROOT / "source" / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_fresh_payload_bytes_and_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 81
    assert len({(r["world"], r["seed"]) for r in rows}) == 9
    max_diff = 0.0
    for row in rows:
        path = ROOT / "artifacts" / "payloads" / f'{row["world"]}_{row["seed"]}_{row["method"]}_N{row["n"]}.pt'
        blob = path.read_bytes()
        assert len(blob) == int(row["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest() == row["hash"]
        obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        out = io.BytesIO(); torch.save(obj, out)
        assert out.getvalue() == blob
        kb, vb, ka, va, keys, vals = RUN.make_world(int(row["world"]), int(row["seed"]))
        kh, vh = RUN.decode(row["method"], blob)
        met = RUN.metrics(keys[:int(row["n"])], vals[:int(row["n"])], kh, vh)
        for actual, col in zip(met, ("edit_efficacy_nrmse_mean", "edit_efficacy_nrmse_max", "specificity_drift_mean", "specificity_drift_max")):
            diff = abs(actual - float(row[col])); max_diff = max(max_diff, diff)
            assert diff < 1e-12
    assert max_diff < 1e-12


def test_registered_n64_gate_and_amortization_boundary():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    for world in (47110, 47111, 47112):
        for seed in (0, 1, 2):
            group = [r for r in rows if int(r["world"]) == world and int(r["seed"]) == seed and int(r["n"]) == 64]
            by = {r["method"]: r for r in group}
            assert int(by["mirror"]["payload_bytes"]) <= 0.8 * int(by["rome"]["payload_bytes"])
            assert float(by["mirror"]["edit_efficacy_nrmse_max"]) <= 1e-5
            assert float(by["mirror"]["specificity_drift_max"]) <= 1e-5
            assert int(by["mirror"]["payload_bytes"]) < int(by["generic_coeff"]["payload_bytes"])
    for r in rows:
        if int(r["n"]) == 1 and r["method"] == "mirror":
            assert int(r["payload_bytes"]) > 0
