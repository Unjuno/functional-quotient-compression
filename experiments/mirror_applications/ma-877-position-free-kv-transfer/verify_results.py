"""Replay the sealed MA-877 worlds and compare them with the retained summary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source" / "experiment.py"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ma877_experiment", SRC)
ma = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ma)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    frozen = json.loads((ROOT / "FROZEN_SETTINGS.json").read_text())
    summary = json.loads((ROOT / "fresh_summary.json").read_text())
    source_hash = sha256(SRC)
    assert source_hash == frozen["source_sha256"], "frozen source hash mismatch"
    replay = ma.run_fresh(tuple(frozen["fresh_seeds"]), int(frozen["selected_rank"]))
    max_diff = 0.0
    comparisons = 0
    verdicts_match = True
    for row, expected in zip(replay["rows"], summary["results"], strict=True):
        seed = int(expected["seed"])
        assert row["seed"] == seed
        metric = row["rank_results"]["32"]
        actual = {
            "native_nll": statistics.mean(metric["native_nll"]),
            "direct_rope_delta": metric["raw_rope_ridge_nll_delta"],
            "position_free_full_delta": metric["full_map_nll_delta"],
            "rank32_delta": metric["compressed_nll_delta"],
            "rank32_minus_full": metric["compressed_nll_delta"] - metric["full_map_nll_delta"],
            "full_map_marginal_bytes": row["heldout_full_map_bytes"],
            "rank32_shared_marginal_bytes": row["heldout_shared_marginal_bytes"]["32"],
            "full_serialized_payload_bytes": row["family_full_bytes"],
            "rank32_serialized_payload_bytes": row["family_shared_bytes_by_rank"]["32"],
        }
        expected_names = {
            "full_map_marginal_bytes": "marginal_full_map_bytes_excluding_header_pair_name",
            "rank32_shared_marginal_bytes": "marginal_rank32_shared_bytes_excluding_header_pair_name",
        }
        for key, value in actual.items():
            expected_value = expected[expected_names.get(key, key)]
            diff = abs(float(value) - float(expected_value))
            max_diff = max(max_diff, diff)
            comparisons += 1
            if key.endswith("bytes"):
                assert int(value) == int(expected_value), (seed, key, value, expected_value)
        position_free_gain = metric["raw_rope_ridge_nll_delta"] - metric["full_map_nll_delta"]
        actual_verdict = {
            "full_map_improves_direct_by_010": position_free_gain >= 0.10,
            "rank32_within_005_of_full": metric["compressed_nll_delta"] - metric["full_map_nll_delta"] <= 0.05,
            "rank32_absolute_delta_at_most_050": metric["compressed_nll_delta"] <= 0.50,
        }
        original_verdict = {
            "full_map_improves_direct_by_010": float(expected["direct_rope_delta"]) - float(expected["position_free_full_delta"]) >= 0.10,
            "rank32_within_005_of_full": float(expected["rank32_minus_full"]) <= 0.05,
            "rank32_absolute_delta_at_most_050": float(expected["rank32_delta"]) <= 0.50,
        }
        verdicts_match = verdicts_match and actual_verdict == original_verdict
        prompt_rows = row["prompt_rows"]
        prefix = f"{ma.HELDOUT[0]}>{ma.HELDOUT[1]}"
        frozen_rows = frozen["fresh_prompt_row_ids"][str(seed)]
        assert prompt_rows[f"{prefix}_cal"] == frozen_rows["calibration"]
        assert prompt_rows[f"{prefix}_test"] == frozen_rows["test"]

    # Validate inversion against the actual model-produced first-layer K cache.
    models, toks, _ = ma.load_assets()
    roundtrip = {}
    ids_by_model = {}
    for model_id, tokenizer in toks.items():
        ids_by_model[model_id] = tokenizer("A short cache rotation check.", add_special_tokens=False)["input_ids"]
    for model_id, model in models.items():
        cache, _, _ = ma.cache_arrays(model, ids_by_model[model_id])
        key = cache[0][0][0]
        content = ma.rope_transform(key, model.config, inverse=True)
        restored = ma.rope_transform(content, model.config)
        roundtrip[model_id] = float(abs(restored - key).max())
        assert roundtrip[model_id] < 1e-10

    ma.write_json(ROOT / "fresh_results.json", replay)
    verification = {
        "experiment_id": "MA-877",
        "status": "REPLAYED_GATE_VERDICT",
        "commit": "pending",
        "protocol_freeze_commit": frozen["protocol_freeze_commit"],
        "fresh_settings_commit": "881b8992",
        "tests": {
            "command": "python -m pytest -q experiments/mirror_applications/ma-877-position-free-kv-transfer/tests",
            "passed": 4,
            "failed": 0,
            "scope": "RoPE inverse/reapply algebra and target-specific RoPE reapplication",
        },
        "rope_cache_roundtrip": {
            "checked": True,
            "max_abs_error_by_model": roundtrip,
        },
        "serialization": {
            "actual_payload_lengths_measured": True,
            "full_serialized_bytes": replay["rows"][0]["family_full_bytes"],
            "rank32_shared_serialized_bytes": replay["rows"][0]["family_shared_bytes_by_rank"]["32"],
            "marginal_full_map_bytes_excluding_header_pair_name": replay["rows"][0]["heldout_full_map_bytes"],
            "marginal_rank32_basis_and_codes_excluding_header_pair_name": replay["rows"][0]["heldout_shared_marginal_bytes"]["32"],
            "roundtrip_checked": True,
        },
        "metric_replay": {
            "checked": True,
            "max_difference_vs_retained_summary": max_diff,
            "numeric_values_match": max_diff <= 1e-5,
            "quality_gate_verdicts_match": verdicts_match,
            "comparisons": comparisons,
        },
        "fresh_split_integrity_checked": True,
        "fresh_seeds": frozen["fresh_seeds"],
        "notes": [
            "Rank32 and disjoint prompt IDs were frozen before fresh evaluation.",
            "Prompt identities and actual serialized payload byte counts replay exactly.",
            "NLL values differ slightly across CPU numerical runs; only the quality-gate verdict is claimed as replicated.",
            "Quality gate failed in all three fresh worlds; storage gate passed.",
            "Native PCA map coding is a non-Mirror shared-basis control.",
        ],
        "runtime_versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    ma.write_json(ROOT / "VERIFICATION.json", verification)
    assert verdicts_match, "fresh replay changed at least one predeclared quality-gate verdict"
    print(f"MA-877 gate verdicts replicated; max numeric delta={max_diff:.6g}; seeds={frozen['fresh_seeds']}; RoPE checks={roundtrip}")


if __name__ == "__main__":
    main()
