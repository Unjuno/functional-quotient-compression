"""Recheck MA-879 prompt isolation, replay identity, payload and preregistered gates."""
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "experiment.py"
spec = importlib.util.spec_from_file_location("ma879_verify", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = json.loads((ROOT / "fresh_results.json").read_text())
initial = json.loads((ROOT / "fresh_screening_run.json").read_text())
assert result["seeds"] == [983, 997, 1009]
assert [x["seed"] for x in result["rows"]] == result["seeds"]
assert [x["seed"] for x in initial["rows"]] == result["seeds"]

for a, b in zip(result["rows"], initial["rows"]):
    assert a["calibration_rows"] == b["calibration_rows"]
    assert a["test_rows"] == b["test_rows"]
    assert a["payload_bytes"] == b["payload_bytes"]
    for key in ("native_nll", "full_ridge_nll", "pca_rank32_nll", "nll_optimized_rank32_nll"):
        assert a["evaluation"][key] == b["evaluation"][key]
    assert not set(a["calibration_rows"]) & set(a["test_rows"])

models, tokenizers, _ = module.load_assets()
cal_texts, test_texts = [], []
for seed in result["seeds"]:
    cal_texts.extend(x["text"] for x in module.select_prompts("validation", tokenizers, seed, 12, "calibration"))
    test_texts.extend(x["text"] for x in module.select_prompts("test", tokenizers, seed, 16, "test"))
assert len(cal_texts) == len(set(cal_texts)) == 36
assert len(test_texts) == len(set(test_texts)) == 48
assert not set(cal_texts) & set(test_texts)

for row in result["rows"]:
    ev, sizes = row["evaluation"], row["payload_bytes"]
    assert sizes["nll_optimized_rank32"] == 573125
    assert sizes["pca_rank32"] == 573125
    assert sizes["full_weighted_ridge"] == 2996165
    gain = statistics.mean(ev["pca_rank32_nll"]) - statistics.mean(ev["nll_optimized_rank32_nll"])
    delta_native = statistics.mean(ev["nll_optimized_rank32_nll"]) - statistics.mean(ev["native_nll"])
    gap_full = statistics.mean(ev["nll_optimized_rank32_nll"]) - statistics.mean(ev["full_ridge_nll"])
    byte_pass = sizes["nll_optimized_rank32"] <= 0.5 * sizes["full_weighted_ridge"]
    assert byte_pass
    assert not (gain >= 0.10 and delta_native <= 0.50 and gap_full <= 0.50)

print("MA-879 verification PASS: 36/36 calibration and 48/48 test prompts unique and split-disjoint; replay NLL/prompt/byte identity exact; byte gate passes and preregistered quality gates fail in all three seeds.")
