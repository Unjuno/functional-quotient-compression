#!/usr/bin/env python3
"""Deterministic checks for MA-878 frozen artifacts and gates."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
locked = json.loads((ROOT / "FROZEN_SETTINGS.json").read_text())
fresh = json.loads((ROOT / "fresh_results.json").read_text())
assert locked["fresh_accessed"] is False
assert fresh["seeds"] == protocol["fresh"]["seeds"] == [983, 997, 1009]
assert fresh["rank"] == locked["selected_rank"] == 32
assert len(fresh["rows"]) == 3
all_cal, all_test = [], []
per_seed = []
for row in fresh["rows"]:
    assert len(row["calibration_rows"]) == 12 and len(row["test_rows"]) == 16
    assert len(set(row["calibration_rows"])) == 12 and len(set(row["test_rows"])) == 16
    assert not set(row["calibration_rows"]) & set(row["test_rows"])
    all_cal.extend(row["calibration_rows"]); all_test.extend(row["test_rows"])
    x = row["rank_results"]["32"]
    assert x["prompt_rows"] == row["test_rows"]
    assert abs(x["matched_gain_vs_fixed"] - (sum(x["fixed_spread_nll"])/16 - sum(x["matched_full_nll"])/16)) < 1e-9
    b = row["payload_bytes_by_rank"]["32"]
    assert b["attention_matched_full"] == 2_996_165
    assert b["attention_matched_rank"] == 573_125
    per_seed.append({"seed": row["seed"], "matched_gain_vs_fixed": x["matched_gain_vs_fixed"],
                     "rank_gap_vs_full": x["rank_gap_vs_full"], "rank_delta_vs_native": x["matched_rank_delta"],
                     "bytes_saved_fraction": 1-b["attention_matched_rank"]/b["attention_matched_full"]})
assert len(set(all_cal)) == len(all_cal) and len(set(all_test)) == len(all_test)
assert set(all_cal).isdisjoint(all_test)
success = all(x["matched_gain_vs_fixed"] >= 0.10 and x["rank_gap_vs_full"] <= 0.05 and
              x["rank_delta_vs_native"] <= 0.50 and x["bytes_saved_fraction"] >= 0.50 for x in per_seed)
assert not success
print(json.dumps({"status": "FAIL", "fresh_prompt_identity_checks": "PASS", "serialized_byte_checks": "PASS",
                  "gate": "FAIL", "per_seed": per_seed}, indent=2, sort_keys=True))
