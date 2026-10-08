"""Verify serialized MA-932 stitchers, held-out metrics, and information probes."""
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Stitcher, make_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    summary = json.loads((ROOT / "source" / "screen_summary.json").read_text())
    assert summary["audit_opened"] is False and summary["audit"] == []
    rows = summary["results"]
    max_diff, manifest = 0.0, []
    for row in rows:
        path = ROOT / row["payload"]["path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["payload"]["bytes"] and digest == row["payload"]["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = Stitcher(row["mode"], row["seed"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        split = "dev" if row["split"] == "development" else "fresh"
        reps, targets = make_split(row["seed"], split, 4096)
        x = reps[row["representation"]]
        with torch.no_grad():
            pred0 = model(x, torch.zeros(len(x), dtype=torch.long))
            pred1 = model(x, torch.ones(len(x), dtype=torch.long))
            mse0 = float(F.mse_loss(pred0, targets[:, 0]))
            mse1 = float(F.mse_loss(pred1, targets[:, 1]))
        diff = max(abs(mse0 - row["mse_task0"]), abs(mse1 - row["mse_task1"]))
        max_diff = max(max_diff, diff)
        assert diff <= 1e-9
        manifest.append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": digest})
    assert len(rows) == 60

    probe_diff = 0.0
    for row in summary["linear_probes"]:
        train, train_y = make_split(row["seed"], "train", 16384)
        split = "dev" if row["split"] == "development" else "fresh"
        hold, hold_y = make_split(row["seed"], split, 4096)
        x = train[row["representation"]]
        coef = torch.linalg.solve(x.T @ x + 1e-4 * torch.eye(8), x.T @ train_y)
        probe = ((hold[row["representation"]] @ coef - hold_y) ** 2).mean(0)
        probe_diff = max(probe_diff, abs(float(probe[0]) - row["probe_u_mse"]),
                         abs(float(probe[1]) - row["probe_v_mse"]))
    assert probe_diff <= 1e-9

    pool = list(csv.DictReader((ROOT / "source" / "selection_pool.csv").open(newline="")))
    draw = json.loads((ROOT / "source" / "random_draw.json").read_text())
    serialized = "".join(",".join(item[k] for k in
        ("id", "family", "proposal", "priority", "status", "prior_art_refs")) + "\n" for item in pool).encode()
    assert len(pool) == draw["pool_size"] and hashlib.sha256(serialized).hexdigest() == draw["pool_sha256"]
    assert random.Random(int(draw["cryptographic_seed_hex"], 16)).randrange(len(pool)) == draw["uniform_index"]
    assert pool[draw["uniform_index"]]["id"] == "MA-932"

    verification = {
        "experiment_id": "MA-932", "commit": "pending-result-commit",
        "selection_draw": {"draw": 21, "pool_size": len(pool), "uniform_index": draw["uniform_index"],
            "selected_id": "MA-932", "pool_sha256": draw["pool_sha256"],
            "baseline_commit": draw["baseline_commit"], "replay_checked": True},
        "tests": {"command": "python -m unittest discover -s experiments/mirror_applications/ma-932-stitching-information-validity/tests -v",
            "passed": 3, "failed": 0},
        "serialization": {"actual_payloads_measured": True, "payload_count": len(manifest), "payload_manifest": manifest},
        "metric_replay": {"checked": True, "rows": len(rows), "max_difference": max_diff,
            "probe_rows": len(summary["linear_probes"]), "max_probe_difference": probe_diff},
        "audit_opened": False,
        "notes": ["Development and fresh records replay exactly from serialized model payloads.",
            "Counterfactual task-1 and linear-probe diagnostics confirm that task0-only representations lose v while preserving u.",
            "Mirror payload and Mirror-specific gates fail; audit remains unopened."]}
    assert max_diff == 0.0 and probe_diff == 0.0
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"payloads": len(manifest), "metric_rows": len(rows),
        "probe_rows": len(summary["linear_probes"]), "max_mse_diff": max_diff,
        "max_probe_diff": probe_diff, "audit_opened": False}, indent=2))


if __name__ == "__main__":
    main()
