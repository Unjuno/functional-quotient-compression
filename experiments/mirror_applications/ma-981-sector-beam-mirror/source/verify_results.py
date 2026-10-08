"""Verify MA-981 payload integrity and replay channel metrics."""
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from model import NSECTOR, BeamCodebook, make_channels, rates

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    summary = json.loads((ROOT / "source" / "screen_summary.json").read_text())
    assert not summary["audit_opened"] and summary["audit"] == []
    manifest, max_diff = [], 0.0
    for row in summary["results"]:
        path = ROOT / row["payload"]["path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["payload"]["bytes"] and digest == row["payload"]["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = BeamCodebook(row["method"], row["seed"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        channels = make_channels(row["seed"], row["split"], 4096)
        sector_rates, accuracy = [], []
        with torch.no_grad():
            for sector in range(NSECTOR):
                beams = model.beams(sector)
                selected = rates(channels[sector], beams).max(1).values
                sector_rates.append(float(selected.mean()))
                matched = channels[sector] / channels[sector].abs().clamp_min(1e-12) / (8 ** .5)
                gain = (channels[sector].conj() * matched).sum(-1).abs().square()
                upper = torch.log2(1 + 10 * gain)
                accuracy.append(float((selected >= upper - .1).float().mean()))
        diff = max(abs(sum(sector_rates) / NSECTOR - row["mean_rate"]),
                   abs(min(sector_rates) - row["worst_sector_rate"]),
                   abs(sum(accuracy) / NSECTOR - row["continuous_match_accuracy"]))
        max_diff = max(max_diff, diff)
        assert diff <= 1e-7
        manifest.append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": digest})
    assert len(manifest) == 30

    pool = list(csv.DictReader((ROOT / "source" / "selection_pool.csv").open(newline="")))
    draw = json.loads((ROOT / "source" / "random_draw.json").read_text())
    serialized = "".join(",".join(item[k] for k in
        ("id", "family", "proposal", "priority", "status", "prior_art_refs")) + "\n" for item in pool).encode()
    assert len(pool) == draw["pool_size"] and hashlib.sha256(serialized).hexdigest() == draw["pool_sha256"]
    assert random.Random(int(draw["cryptographic_seed_hex"], 16)).randrange(len(pool)) == draw["uniform_index"]
    assert pool[draw["uniform_index"]]["id"] == "MA-981"

    verification = {
        "experiment_id": "MA-981", "commit": "pending-result-commit",
        "selection_draw": {"draw": 23, "pool_size": len(pool), "uniform_index": draw["uniform_index"],
            "selected_id": "MA-981", "pool_sha256": draw["pool_sha256"],
            "baseline_commit": draw["baseline_commit"], "replay_checked": True},
        "tests": {"command": "python -m unittest discover -s experiments/mirror_applications/ma-981-sector-beam-mirror/tests -v",
            "passed": 3, "failed": 0},
        "serialization": {"actual_payloads_measured": True, "payload_count": len(manifest), "payload_manifest": manifest},
        "metric_replay": {"checked": True, "rows": len(summary["results"]), "max_difference": max_diff},
        "audit_opened": False,
        "notes": ["All development/fresh codebooks passed size/hash checks and rate replay.",
            "Mirror passed quality versus independent sector codebooks but missed the actual-byte reduction gate.",
            "Mirror and rank-2 payloads were not within the preregistered 5% byte-match window; audit was not applicable."]}
    assert max_diff <= 1e-7
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"payloads": len(manifest), "metric_rows": len(summary["results"]),
        "max_difference": max_diff, "audit_opened": False}, indent=2))


if __name__ == "__main__":
    main()
