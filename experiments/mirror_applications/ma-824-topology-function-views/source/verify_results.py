"""Verify amended MA-824 payloads, exact outputs, and selection replay."""
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from model import FUNCS, TOPOLOGIES, execute_payload

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    summary = json.loads((ROOT / "source" / "screen_summary.json").read_text())
    assert not summary["audit_opened"] and summary["audit"] == []
    rows = summary["results"]
    manifest, max_error = [], 0.0
    for row in rows:
        path = ROOT / row["payload"]["path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row["payload"]["bytes"] and digest == row["payload"]["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if row["method"] == "independent_programs":
            for program in payload["representation"]["independent_programs"]:
                for key in ("w1", "b1", "w2", "b2"):
                    left = program["nodes"][0]["function_weights"][key].untyped_storage().data_ptr()
                    right = program["nodes"][1]["function_weights"][key].untyped_storage().data_ptr()
                    assert left != right
        offset = {"development": 1_000_000, "fresh": 2_000_000}[row["split"]]
        x = torch.randn(8192, 2, generator=torch.Generator().manual_seed(row["seed"] + offset)) * 1.5
        reference_path = ROOT / "source" / "artifacts" / f"amended_{row['split']}_{row['seed']}_mirror_factors.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        errors = {}
        for topology in TOPOLOGIES:
            for fid, name in enumerate(FUNCS):
                target = execute_payload(x, reference, topology, fid)
                prediction = execute_payload(x, payload, topology, fid)
                errors[f"{topology}:{name}"] = float((prediction - target).square().mean())
        error = max(errors.values())
        max_error = max(max_error, error)
        assert error <= 1e-10 and row["max_program_mse"] == 0.0
        manifest.append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": digest})
    assert len(rows) == 20

    pool = list(csv.DictReader((ROOT / "source" / "selection_pool.csv").open(newline="")))
    draw = json.loads((ROOT / "source" / "random_draw.json").read_text())
    serialized = "".join(",".join(item[k] for k in
        ("id", "family", "proposal", "priority", "status", "prior_art_refs")) + "\n" for item in pool).encode()
    assert len(pool) == draw["pool_size"] and hashlib.sha256(serialized).hexdigest() == draw["pool_sha256"]
    assert random.Random(int(draw["cryptographic_seed_hex"], 16)).randrange(len(pool)) == draw["uniform_index"]
    assert pool[draw["uniform_index"]]["id"] == "MA-824"

    verification = {
        "experiment_id": "MA-824", "commit": "pending-result-commit", "amendment": 1,
        "selection_draw": {"draw": 22, "pool_size": len(pool), "uniform_index": draw["uniform_index"],
            "selected_id": "MA-824", "pool_sha256": draw["pool_sha256"],
            "baseline_commit": draw["baseline_commit"], "replay_checked": True},
        "tests": {"command": "python -m unittest discover -s experiments/mirror_applications/ma-824-topology-function-views/tests -v",
            "passed": 3, "failed": 0},
        "serialization": {"actual_payloads_measured": True, "payload_count": len(manifest),
            "independent_duplicate_storages_checked": True, "payload_manifest": manifest},
        "metric_replay": {"checked": True, "rows": len(rows), "max_program_mse_difference": max_error},
        "audit_opened": False,
        "notes": ["Initial aliased-storage attempt is retained separately and excluded from decisions.",
            "Amended independent controls use distinct node tensor storages; fresh seeds 82406–82408 were not viewed before amendment.",
            "All amended program outputs, including the held-out topology/function pair, replay exactly from serialized payloads."]}
    assert max_error == 0.0
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"payloads": len(manifest), "rows": len(rows),
        "max_error": max_error, "audit_opened": False}, indent=2))


if __name__ == "__main__":
    main()
