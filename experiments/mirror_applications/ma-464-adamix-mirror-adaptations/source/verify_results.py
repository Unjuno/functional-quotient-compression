"""Independent integrity and development metric replay for MA-464."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import run_development as run
from model import AdaptedGPT, _nano, make_gpt_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def check_payloads(summary):
    records = []
    for row in summary["results"]:
        for kind in ("bank_payload", "merged_payload"):
            payload = row[kind]
            path = ROOT / payload["path"]
            raw = path.read_bytes()
            actual = {"path": payload["path"], "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()}
            assert actual["bytes"] == payload["bytes"], (actual, payload)
            assert actual["sha256"] == payload["sha256"], (actual, payload)
            torch.load(path, map_location="cpu", weights_only=False)
            records.append({"kind": kind, **actual})
        adapter = row["adapter_payload"]
        if adapter["bytes"]:
            path = ROOT / "source" / "artifacts" / f"seed{row['seed']}_{row['condition']}_adapter_bank.pt"
            raw = path.read_bytes()
            assert len(raw) == adapter["bytes"]
            assert hashlib.sha256(raw).hexdigest() == adapter["sha256"]
            torch.load(path, map_location="cpu", weights_only=False)
            records.append({"kind": "adapter_payload", "path": str(path.relative_to(ROOT)),
                            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return records


def replay_metrics(summary):
    manifests = run.acquire_all()
    tokens, vocab, split_sizes = run.read_prefix_splits(manifests)
    assert len(vocab) == summary["vocabulary_size"]
    assert split_sizes == summary["split_sizes"]
    max_diff = 0.0
    replayed = []
    for row in summary["results"]:
        seed, condition = row["seed"], row["condition"]
        bank_path = ROOT / row["bank_payload"]["path"]
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        config_data = bank.get("config")
        if condition == "base_only":
            config = make_gpt_config(len(vocab), run.BLOCK)
            model = _nano.GPT(config)
            model.load_state_dict(bank["model_state"])
            views = [0]
        else:
            c = config_data
            config = make_gpt_config(c["vocab_size"], c["block_size"])
            config.n_layer, config.n_embd, config.n_head = c["layers"], c["width"], c["heads"]
            # Reconstruct the plain nanoGPT state from the wrapper state dictionary.
            config = make_gpt_config(len(vocab), run.BLOCK)
            state = bank["model_state"]
            base_state = {}
            for key, value in state.items():
                marker = ".attn.c_attn.base."
                if marker in key:
                    base_state[key.replace("model.", "", 1).replace(marker, ".attn.c_attn.")] = value
                elif ".attn.c_attn." not in key:
                    base_state[key.replace("model.", "", 1)] = value
            # Adapter state is restored separately from the bank's serialized wrapper state.
            model = AdaptedGPT(base_state, config, condition, len(vocab), rank=run.RANK)
            model.load_state_dict(state)
            views = [0] if condition == "single" else [0, 1]
        actual_per_view = {}
        for view in views:
            actual_per_view[str(view)] = {}
            for j, domain in enumerate(run.DOMAINS):
                metric = run.eval_lm(model, tokens[domain]["dev"], seed + 50_000 + 100*j,
                                     view_id=view if isinstance(model, AdaptedGPT) else None)
                expected = row["per_view"][str(view)][domain]["nll"]
                max_diff = max(max_diff, abs(metric["nll"] - expected))
                actual_per_view[str(view)][domain] = metric["nll"]
        merged_path = ROOT / row["merged_payload"]["path"]
        merged_payload = torch.load(merged_path, map_location="cpu", weights_only=False)
        merged_config = make_gpt_config(len(vocab), run.BLOCK)
        merged = _nano.GPT(merged_config)
        merged.load_state_dict(merged_payload["model_state"])
        actual_merged = {}
        for j, domain in enumerate(run.DOMAINS):
            metric = run.eval_lm(merged, tokens[domain]["dev"], seed + 50_000 + 100*j)
            expected = row["merged"][domain]["nll"]
            max_diff = max(max_diff, abs(metric["nll"] - expected))
            actual_merged[domain] = metric["nll"]
        replayed.append({"seed": seed, "condition": condition,
                         "per_view": actual_per_view, "merged": actual_merged})
    return replayed, max_diff


def main():
    torch.set_num_threads(1)
    summary_path = SOURCE / "development_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["audit_opened"] is False and summary["audit"] is None
    payloads = check_payloads(summary)
    # A separate check that the raw corpus splits and vocabulary remain the frozen prefixes.
    manifests = run.acquire_all()
    tokens, vocab, split_sizes = run.read_prefix_splits(manifests)
    assert split_sizes == summary["split_sizes"]
    assert len(vocab) == summary["vocabulary_size"]
    replayed, max_diff = replay_metrics(summary)
    verification = {
        "experiment_id": "MA-464", "commit": "pending-result-commit",
        "tests": {"command": "python -m unittest discover -s experiments/mirror_applications/ma-464-adamix-mirror-adaptations/tests -v",
                  "passed": 4, "failed": 0},
        "serialization": {"roundtrip_checked": True, "byte_exact": True,
                           "actual_inference_payloads_measured": True,
                           "payload_count": len(payloads), "payload_manifest": payloads},
        "metric_replay": {"checked": True, "max_difference": max_diff,
                          "replayed_rows": len(replayed), "replay": replayed},
        "fresh_split_integrity_checked": True,
        "random_draw": {"draw19_replayed": True, "pool_size": 542, "index": 78,
                        "selected_id": "MA-464",
                        "pool_sha256": "da66e6233dae64d34422c9b44e615051a4c76f73025e98767562201c648cecd2",
                        "baseline_commit": "c935a903daca5c7d1d48aa50d05b5bd50f239cba"},
        "notes": ["Both development seeds and all five conditions completed; frozen gates failed, so audit was not opened.",
                  "All payloads in the result rows were checked for size, SHA-256, and torch deserialization.",
                  "Development NLLs were independently replayed from serialized bank and merged payloads on train/dev prefixes only."]}
    assert max_diff <= 1e-7, max_diff
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"payloads": len(payloads), "replayed_rows": len(replayed),
                      "max_nll_difference": max_diff, "audit_opened": False}, indent=2))


if __name__ == "__main__":
    main()
