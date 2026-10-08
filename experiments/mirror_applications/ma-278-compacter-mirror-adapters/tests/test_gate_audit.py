import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_development_gate_and_scalar_equivalence_audit():
    with (ROOT / "RESULTS_CORE.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert {r["world_or_seed"] for r in rows} == {"27800", "27801"}
    for world in {"27800", "27801"}:
        for lr in {"0.003", "0.01"}:
            records = {r["method"]: r for r in rows if r["world_or_seed"] == world and r["status_note"] == f"lr={lr}"}
            assert set(records) == {"tied", "compacter", "scalar", "mirror", "independent"}
            assert records["mirror"]["serialized_bytes"] == records["scalar"]["serialized_bytes"] == "2525"
            assert float(records["mirror"]["primary_value"]) > float(records["scalar"]["primary_value"])
            assert float(records["mirror"]["primary_value"]) > 1.10 * float(records["independent"]["primary_value"])
