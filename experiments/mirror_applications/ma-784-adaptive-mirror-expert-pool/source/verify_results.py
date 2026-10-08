"""Replay development NLL and verify actual serialized payloads; never reads audit."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))
from model import SmallGPT
from acquire_corpus import acquire
from run_development import ARTIFACTS, BATCH, EVAL_BATCHES, eval_model, make_config, sample_positions, split_and_encode


def main():
    torch.set_num_threads(1)
    summary = json.loads((SOURCE / "development_summary.json").read_text())
    manifest = summary["corpus"]
    assert acquire()["sha256"] == manifest["sha256"]
    assert summary["gates"]["development_gate_pass"] is False
    assert summary["audit"] is None and summary["split"]["audit_opened"] is False
    train, dev, vocab = split_and_encode(dict(manifest))
    assert train.numel() == summary["split"]["train_characters"]
    assert dev.numel() == summary["split"]["dev_characters"]
    assert summary["corpus"]["audit_bytes_not_read"] > 0
    max_diff = 0.0
    checked = 0
    for row in summary["results"]:
        seed, condition = row["seed"], row["condition"]
        path = ARTIFACTS / f"seed{seed}_{condition}_inference.pt"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["inference_payload"]["bytes"]
        assert digest == row["inference_payload"]["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["condition"] == condition and payload["seed"] == seed
        assert len(payload["vocabulary"]) == len(vocab)
        assert payload["corpus_sha256"] == manifest["sha256"]
        model = SmallGPT(make_config(), condition, len(vocab))
        model.load_state_dict(payload["model_state"])
        positions = sample_positions(dev.numel(), BATCH * EVAL_BATCHES, seed + 30000)
        replay = eval_model(model, dev, positions)
        diff = abs(replay["nll"] - row["nll"])
        max_diff = max(max_diff, diff)
        assert diff <= 1e-12, (seed, condition, diff)
        checked += 1
    result = {"payload_count": checked, "payload_bytes_and_hashes_exact": True,
              "development_metric_replay": True, "max_abs_nll_difference": max_diff,
              "train_dev_split_reloaded": True, "audit_opened": False,
              "audit_semantic_read": False,
              "note": "Corpus SHA-256 was streamed for provenance. The final 10% was not decoded, tokenized, evaluated, or used for selection."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
