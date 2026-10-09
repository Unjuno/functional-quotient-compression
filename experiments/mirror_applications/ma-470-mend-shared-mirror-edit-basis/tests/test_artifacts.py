import csv
import hashlib
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_every_fresh_payload_roundtrips_byte_exactly_and_metrics_replay():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    assert len(rows) == 135
    assert len({(r["world"], r["seed"]) for r in rows}) == 9
    max_error = 0.0
    for row in rows:
        payload = ROOT / "artifacts" / "payloads" / f'{row["world"]}_{row["seed"]}_{row["method"]}_N{row["n"]}.pt'
        payload = payload.read_bytes()
        assert len(payload) == int(row["payload_bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["hash"]
        decoded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        buf = io.BytesIO()
        torch.save(decoded, buf)
        assert buf.getvalue() == payload
        max_error = max(max_error, abs(len(payload) - int(row["payload_bytes"])))
    assert max_error == 0


def test_frozen_gates_and_off_orbit_private_boundary():
    rows = list(csv.DictReader((ROOT / "RESULTS_CORE.csv").open()))
    for world in (47010, 47011, 47012):
        for seed in (0, 1, 2):
            pair = [r for r in rows if int(r["world"]) == world and int(r["seed"]) == seed and int(r["n"]) == 64]
            by = {r["method"]: r for r in pair}
            assert float(by["mirror_private"]["edit_nrmse"]) <= 1.1 * float(by["mend"]["edit_nrmse"]) + 1e-6
            assert int(by["mirror_private"]["payload_bytes"]) <= 0.75 * int(by["mend"]["payload_bytes"])
            assert float(by["mirror"]["edit_nrmse"]) > 0.2
            assert float(by["mirror_private"]["locality_drift"]) < 1e-5
