"""Integrity and preregistered-gate checks for the MA-881 saved audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "fresh_results.json").read_text())
assert data["selected_rank"] == 32
assert [r["seed"] for r in data["rows"]] == [983, 997, 1009]
gates = []
for row in data["rows"]:
    shared, full, single = row["shared_mot"], row["full_mot"], row["single_ridge"]
    payload = ROOT / "artifacts" / f"shared_mot_payload_seed_{row['seed']}.bin"
    raw = payload.read_bytes()
    assert len(raw) == row["payload_bytes"]["shared_view_mot"]
    assert hashlib.sha256(raw).hexdigest() == row["payload_sha256"]
    ids = row["prompt_rows"]
    assert len(set(ids["expert0"] + ids["expert1"] + ids["router"])) == 12
    assert not (set(ids["calibration"]) & set(ids["test"]))
    assert abs(shared["native_mean_nll"] - single["native_mean_nll"]) < 1e-9
    q = (shared["mot_mean_nll"] <= full["mot_mean_nll"] + 0.5
         and shared["mot_mean_nll"] <= single["single_ridge_mean_nll"] + 1.0
         and shared["mot_mean_nll"] <= shared["native_mean_nll"] + 1.0)
    s = row["bytes_saved_fraction"] >= 0.5
    gates.append({"seed": row["seed"], "quality_pass": q, "storage_pass": s,
                  "delta_full": shared["mot_mean_nll"] - full["mot_mean_nll"],
                  "delta_single": shared["mot_mean_nll"] - single["single_ridge_mean_nll"],
                  "delta_native": shared["mot_mean_nll"] - shared["native_mean_nll"],
                  "bytes": row["payload_bytes"], "saved_fraction": row["bytes_saved_fraction"]})
assert all(g["storage_pass"] for g in gates)
assert not any(g["quality_pass"] for g in gates)
print(json.dumps({"experiment_id": "MA-881", "integrity": "PASS", "gates": gates}, indent=2))
