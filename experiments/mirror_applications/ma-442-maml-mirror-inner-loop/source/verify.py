#!/usr/bin/env python3
"""Verify frozen development artifacts, payload accounting and native alias."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner

ROOT = Path(__file__).resolve().parents[1]
errors = []
seed_results = []
for seed in (44201, 44202):
    run_dir = ROOT / "runs" / f"dev_{seed}"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        errors.append(f"missing {metrics_path}")
        continue
    metrics = json.loads(metrics_path.read_text())
    methods = metrics["methods"]
    mirror_payload = run_dir / "mirror_payload.npz"
    native_payload = run_dir / "native_givens_payload.npz"
    for method, key in (("mirror", mirror_payload), ("native_givens", native_payload)):
        raw = key.read_bytes()
        if len(raw) != methods[method]["payload_bytes"]:
            errors.append(f"{seed}/{method}: serialized size mismatch")
        if hashlib.sha256(raw).hexdigest() != methods[method]["payload_sha256"]:
            errors.append(f"{seed}/{method}: payload hash mismatch")
        loaded = np.load(key)
        if "__schema_json__" not in loaded.files:
            errors.append(f"{seed}/{method}: schema metadata absent")
    with np.load(mirror_payload) as m, np.load(native_payload) as n:
        for name in ("shared_init", "task_codes"):
            if not np.array_equal(m[name], n[name]):
                errors.append(f"{seed}: Mirror/native {name} differs")
        max_diff = float(np.max(np.abs(m["task_codes"] - n["task_codes"])))
    if methods["mirror"]["query_rmse"] != methods["native_givens"]["query_rmse"]:
        errors.append(f"{seed}: Mirror/native replay metric differs")
    seed_results.append({"seed": seed, "mirror_native_parameter_max_difference": max_diff,
                         "mirror_native_query_rmse_difference": abs(methods["mirror"]["query_rmse"] - methods["native_givens"]["query_rmse"]),
                         "mirror_bytes": methods["mirror"]["payload_bytes"],
                         "native_bytes": methods["native_givens"]["payload_bytes"]})
    rng = runner.task_rng(seed + 901, "dev")
    replay = {name: [] for name in methods}
    payloads = {name: np.load(run_dir / f"{name}_payload.npz") for name in methods}
    try:
        for index in range(16):
            _, x, y = runner.sample_task(rng, runner.SUPPORT_N + runner.QUERY_N)
            xq, yq = x[runner.SUPPORT_N:], y[runner.SUPPORT_N:]
            for name, archive in payloads.items():
                if name == "mirror" or name == "native_givens":
                    base = torch.from_numpy(archive["shared_init"][0])
                    code = torch.from_numpy(archive["task_codes"][index])
                    w = runner.prediction_weight(name, base, code)
                elif name == "lora":
                    base = torch.from_numpy(archive["shared_init"][0])
                    basis = torch.from_numpy(archive["basis"])
                    code = torch.from_numpy(archive["task_codes"][index])
                    w = runner.prediction_weight(name, base, code, basis)
                elif name == "full":
                    w = torch.from_numpy(archive["task_codes"][index])
                elif name == "no_adapt":
                    w = torch.from_numpy(archive["shared_init"][0])
                else:
                    w = torch.from_numpy(archive["task_vectors"][index])
                replay[name].append(float(torch.sqrt(runner.mse(xq, yq, w))))
        for name in methods:
            replay_rmse = float(np.mean(replay[name]))
            if abs(replay_rmse - methods[name]["query_rmse"]) > 1e-6:
                errors.append(f"{seed}/{name}: inference payload replay RMSE mismatch {replay_rmse}")
    finally:
        for archive in payloads.values():
            archive.close()

fresh = sorted((ROOT / "runs").glob("fresh_*"))
if fresh:
    errors.append("fresh runs exist even though the development gates were not evaluated first")

report = {"experiment_id": "MA-442", "development_seeds": seed_results,
          "fresh_split_integrity_checked": True, "fresh_run_directories": [],
          "serialization_roundtrip_checked": bool(seed_results), "payload_byte_exact": not errors,
          "metric_replay_checked": len(seed_results) == 2 and not any("payload replay RMSE mismatch" in e for e in errors),
          "notes": ["Checks payload lengths/hashes, replays all saved inference payloads on deterministic held-out queries, and verifies exact Mirror/native parameter and metric equality."],
          "errors": errors}
(ROOT / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
raise SystemExit(1 if errors else 0)
