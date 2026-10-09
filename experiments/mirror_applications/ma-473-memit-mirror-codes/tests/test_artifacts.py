import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma473_run", ROOT / "source" / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_all_payloads_and_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 144
    assert len({(r["world"], r["seed"]) for r in rows}) == 9
    max_diff = 0.0
    for row in rows:
        method, n = row["method"], int(row["n"])
        bk, bv, ak, av, keys, vals, scales = RUN.make_world(int(row["world"]), int(row["seed"]))
        blob = RUN.payload(method, bk, bv, ak, av, keys, vals, scales, n)
        assert len(blob) == int(row["payload_bytes"])
        assert hashlib.sha256(blob).hexdigest() == row["hash"]
        if method != "no_edit":
            path = ROOT / "artifacts" / "payloads" / f'{row["world"]}_{row["seed"]}_{method}_N{n}.pt'
            assert path.read_bytes() == blob
            obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            b = io.BytesIO(); torch.save(obj, b)
            assert b.getvalue() == blob
        kh, vh = RUN.decode(method, blob)
        eff, effmax, loc, merged = RUN.evaluate(keys[:n], vals[:n], kh, vh, n)
        for actual, col in zip((eff, effmax, loc, merged), ("standalone_efficacy_nrmse_mean", "standalone_efficacy_nrmse_max", "specificity_drift_mean", "merged_bank_error_mean")):
            diff = abs(actual - float(row[col])); max_diff = max(max_diff, diff)
            assert diff < 1e-10
    assert max_diff < 1e-10


def test_n64_storage_gates_and_shared_interference_boundary():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    for world in (47310, 47311, 47312):
        for seed in (0, 1, 2):
            group = [r for r in rows if int(r["world"]) == world and int(r["seed"]) == seed and int(r["n"]) == 64]
            by = {r["method"]: r for r in group}
            assert int(by["mirror"]["payload_bytes"]) <= 0.8 * int(by["memit_factors"]["payload_bytes"])
            assert int(by["mirror"]["payload_bytes"]) <= 0.9 * int(by["generic_coeff"]["payload_bytes"])
            assert float(by["mirror"]["standalone_efficacy_nrmse_max"]) <= 1e-5
            assert float(by["mirror"]["specificity_drift_mean"]) <= 1e-5
            direct = float(by["memit_factors"]["merged_bank_error_mean"])
            mir = float(by["mirror"]["merged_bank_error_mean"])
            assert abs(mir-direct) / max(direct, 1e-9) <= 0.01
    n64 = [float(r["merged_bank_error_mean"]) for r in rows if r["method"] == "mirror" and int(r["n"]) == 64]
    assert sum(n64) / len(n64) > 3.0
