#!/usr/bin/env python3
"""Gauge-invariant leave-one-adapter-out audit for MA-1097.

This is an oracle weight-update reconstruction screen. It does not train a
Mirror code or measure downstream task quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "source" / "artifact_manifest.json"
TASKS = ["tweet_eval_irony", "tweet_eval_emotion", "tweet_eval_hate"]
LAYERS = 12
MODULES = ("query", "value")


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "MA-1097-research/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            path.write_bytes(resp.read())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_bank(cache: Path):
    obj = json.loads(MANIFEST.read_text())
    bank = {}
    source_paths = {}
    for item in obj["tasks"]:
        path = cache / item["file"]
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/adapter_model.safetensors"
        fetch(url, path)
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {item['task']}")
        config_path = cache / f"{item['task']}_adapter_config.json"
        config_url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/adapter_config.json"
        fetch(config_url, config_path)
        if (config_path.stat().st_size != item["adapter_config_size_bytes"]
                or sha256(config_path) != item["adapter_config_sha256"]):
            raise ValueError(f"adapter config hash/size mismatch: {item['task']}")
        source_paths[item["task"]] = path
        tensors = {}
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                tensors[key] = np.asarray(f.get_tensor(key))
        records = {}
        for layer in range(LAYERS):
            for module in MODULES:
                prefix = ("base_model.model.bert.encoder.layer." + str(layer)
                          + ".attention.self." + module)
                akey, bkey = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
                A, B = tensors[akey].astype(np.float64), tensors[bkey].astype(np.float64)
                if A.shape != (1, 768) or B.shape != (768, 1):
                    raise ValueError(f"unexpected factor shape: {item['task']} {prefix}")
                records[(layer, module)] = {"A": A, "B": B, "D": B @ A}
        heads = {k: v for k, v in tensors.items() if ".classifier." in k}
        if len(heads) != 2:
            raise ValueError(f"expected classifier weight+bias in {item['task']}")
        bank[item["task"]] = {"layers": records, "heads": heads}
    return obj, bank, source_paths


def orth_basis(mats, side: str, k: int):
    # Each task contributes one update. SVD uses training-task matrices only.
    xs = [D if side == "out" else D.T for D in mats]
    u, _, _ = np.linalg.svd(np.concatenate(xs, axis=1), full_matrices=False)
    return u[:, :k]


def orth_basis_from_factors(records, side: str, k: int):
    """Exact shared covariance basis from rank-1 B/A factors, without dense SVD."""
    if side == "out":
        x = np.column_stack([r["B"][:, 0] * np.linalg.norm(r["A"]) for r in records])
    elif side == "in":
        x = np.column_stack([r["A"].T[:, 0] * np.linalg.norm(r["B"]) for r in records])
    else:
        raise ValueError("side must be 'out' or 'in'")
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k]


def rel_error(actual, reconstructed):
    den = sum(float(np.sum(x * x)) for x in actual)
    num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(actual, reconstructed))
    return float(np.sqrt(num / max(den, 1e-300)))


def gauge_diagnostic(B: np.ndarray, A: np.ndarray, scale: float = 7.25) -> float:
    """Maximum update/projector drift under an invertible rank-1 gauge."""
    B2, A2 = B * scale, A / scale
    D1, D2 = B @ A, B2 @ A2
    def proj(x):
        q, _ = np.linalg.qr(x, mode="reduced")
        return q @ q.T
    return max(float(np.max(np.abs(D1 - D2))),
               float(np.max(np.abs(proj(B) - proj(B2)))),
               float(np.max(np.abs(proj(A.T) - proj(A2.T)))))


def to_torch(t):
    return torch.from_numpy(np.asarray(t, dtype=np.float32).copy()).contiguous()


def metadata(obj, method, k, fold):
    source_revisions = ";".join(
        f"{x['task']}@{x['revision']}" for x in obj["tasks"])
    return {
        "experiment": "MA-1097",
        "method": method,
        "rank": str(k),
        "heldout_task": TASKS[fold],
        "tasks": ",".join(TASKS),
        "base_model": obj["current_pinned_base"]["repo"],
        "base_revision": obj["current_pinned_base"]["revision"],
        "base_training_revision": "unreported",
        "source_revisions": source_revisions,
        "lora_alpha_over_rank": "1",
        "modules": "12x(query,value)",
        "heads": "task-specific classifier parameters included",
        "code_semantics": "oracle projection of each natural update; no code training",
    }


def serialize_independent(obj, bank, out: Path):
    ts = {}
    for task in TASKS:
        for layer, module in ((l, m) for l in range(LAYERS) for m in MODULES):
            rec = bank[task]["layers"][(layer, module)]
            prefix = f"{task}.layer{layer}.{module}"
            ts[prefix + ".A"] = to_torch(rec["A"])
            ts[prefix + ".B"] = to_torch(rec["B"])
        for key, value in bank[task]["heads"].items():
            ts[f"{task}." + key] = to_torch(value)
    save_file(ts, str(out), metadata={
        "experiment": "MA-1097", "method": "independent_rank1_lora",
        "tasks": ",".join(TASKS), "base_model": obj["current_pinned_base"]["repo"],
        "base_revision": obj["current_pinned_base"]["revision"],
        "base_training_revision": "unreported", "modules": "12x(query,value)",
        "source_revisions": ";".join(f"{x['task']}@{x['revision']}" for x in obj["tasks"]),
        "lora_alpha_over_rank": "1",
        "heads": "task-specific classifier parameters included",
    })


def reconstruct_fold(obj, bank, outdir: Path, fold: int, k: int, method: str):
    held = TASKS[fold]
    train_tasks = [t for t in TASKS if t != held]
    tensors = {}
    actual_by_task = {t: [] for t in TASKS}
    pred_by_task = {t: [] for t in TASKS}
    fit_seconds = 0.0
    decode_macs = 0
    active_macs_per_token = 0
    projector_max = 0.0
    reconstruct_seconds = 0.0

    for layer in range(LAYERS):
        for module in MODULES:
            actual = [bank[t]["layers"][(layer, module)]["D"] for t in TASKS]
            for task, D in zip(TASKS, actual):
                actual_by_task[task].append(D)
            start = time.perf_counter()

            if method == "shared_output_B":
                train_records = [bank[t]["layers"][(layer, module)] for t in train_tasks]
                U = orth_basis_from_factors(train_records, "out", k)
                fit_seconds += time.perf_counter() - start
                start = time.perf_counter()
                tensors[f"basis.layer{layer}.{module}.U"] = to_torch(U)
                for task in TASKS:
                    rec = bank[task]["layers"][(layer, module)]
                    coeff = U.T @ rec["B"]
                    Ahat = rec["A"]
                    Dhat = U @ coeff @ Ahat
                    tensors[f"{task}.layer{layer}.{module}.code"] = to_torch(coeff)
                    tensors[f"{task}.layer{layer}.{module}.private_A"] = to_torch(Ahat)
                    pred_by_task[task].append(Dhat)
                    decode_macs += U.shape[0] * U.shape[1] + U.shape[1]
                    active_macs_per_token = LAYERS * len(MODULES) * (768 + 768)
                    projector_max = max(projector_max, gauge_diagnostic(rec["B"], rec["A"]))
                reconstruct_seconds += time.perf_counter() - start
            elif method == "shared_input_A":
                train_records = [bank[t]["layers"][(layer, module)] for t in train_tasks]
                V = orth_basis_from_factors(train_records, "in", k)
                fit_seconds += time.perf_counter() - start
                start = time.perf_counter()
                tensors[f"basis.layer{layer}.{module}.V"] = to_torch(V)
                for task in TASKS:
                    rec = bank[task]["layers"][(layer, module)]
                    coeff = rec["A"] @ V
                    Bhat = rec["B"]
                    Dhat = Bhat @ coeff @ V.T
                    tensors[f"{task}.layer{layer}.{module}.private_B"] = to_torch(Bhat)
                    tensors[f"{task}.layer{layer}.{module}.code"] = to_torch(coeff)
                    pred_by_task[task].append(Dhat)
                    decode_macs += V.shape[0] * V.shape[1] + V.shape[1]
                    active_macs_per_token = LAYERS * len(MODULES) * (768 + 768)
                    projector_max = max(projector_max, gauge_diagnostic(rec["B"], rec["A"]))
                reconstruct_seconds += time.perf_counter() - start
            elif method == "two_sided_dense_core":
                train_records = [bank[t]["layers"][(layer, module)] for t in train_tasks]
                Us = orth_basis_from_factors(train_records, "out", k)
                Vs = orth_basis_from_factors(train_records, "in", k)
                fit_seconds += time.perf_counter() - start
                start = time.perf_counter()
                tensors[f"basis.layer{layer}.{module}.U"] = to_torch(Us)
                tensors[f"basis.layer{layer}.{module}.V"] = to_torch(Vs)
                for task in TASKS:
                    D = bank[task]["layers"][(layer, module)]["D"]
                    C = Us.T @ D @ Vs
                    Dhat = Us @ C @ Vs.T
                    tensors[f"{task}.layer{layer}.{module}.dense_core"] = to_torch(C)
                    pred_by_task[task].append(Dhat)
                    decode_macs += Us.shape[0] * k * k + k * k * Vs.shape[0]
                    active_macs_per_token = LAYERS * len(MODULES) * (768 * k + k * k + k * 768)
                    projector_max = max(projector_max, gauge_diagnostic(
                        bank[task]["layers"][(layer, module)]["B"],
                        bank[task]["layers"][(layer, module)]["A"]))
                reconstruct_seconds += time.perf_counter() - start
            else:
                raise ValueError(method)

    for task in TASKS:
        for key, value in bank[task]["heads"].items():
            tensors[f"{task}." + key] = to_torch(value)

    serialize_start = time.perf_counter()
    out = outdir / f"{method}_fold{fold}_k{k}.safetensors"
    save_file(tensors, str(out), metadata=metadata(obj, method, k, fold))
    serialize_seconds = time.perf_counter() - serialize_start
    errors = {t: rel_error(actual_by_task[t], pred_by_task[t]) for t in TASKS}
    return {
        "fold": fold, "heldout_task": held, "train_tasks": train_tasks,
        "errors": errors, "heldout_rel_fro_error": errors[held],
        "payload_bytes": out.stat().st_size, "payload_sha256": sha256(out),
        "fit_wall_seconds": fit_seconds, "reconstruct_wall_seconds": reconstruct_seconds,
        "serialize_wall_seconds": serialize_seconds,
        "decode_macs": int(decode_macs), "gauge_max_error": projector_max,
        "active_macs_per_token": int(active_macs_per_token),
        "payload_file": out.name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="directory for pinned public adapter files")
    ap.add_argument("--out", type=Path, required=True, help="directory for temporary serialized payloads/results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    obj, bank, source_paths = load_bank(args.cache)
    source_weights_bytes = sum(p.stat().st_size for p in source_paths.values())
    source_configs_bytes = sum(x["adapter_config_size_bytes"] for x in obj["tasks"])
    independent_path = args.out / "independent_rank1_lora.safetensors"
    t0 = time.perf_counter()
    serialize_independent(obj, bank, independent_path)
    independent_seconds = time.perf_counter() - t0
    independent_bytes = independent_path.stat().st_size
    rows = []
    details = {"scope": "oracle weight-space reconstruction; no downstream task quality",
               "source_artifacts": [{"task": t, "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for t, p in source_paths.items()],
               "source_original_safetensors_bytes": source_weights_bytes,
               "source_adapter_config_bytes": source_configs_bytes,
               "source_payload_bytes_including_configs": source_weights_bytes + source_configs_bytes,
               "independent_bundle_bytes": independent_bytes,
               "independent_bundle_sha256": sha256(independent_path),
               "independent_serialize_seconds": independent_seconds,
               "results": []}
    for fold in range(len(TASKS)):
        for k in (1, 2):
            for method in ("shared_output_B", "shared_input_A", "two_sided_dense_core"):
                item = reconstruct_fold(obj, bank, args.out, fold, k, method)
                item["method"] = method
                item["k"] = k
                item["payload_ratio_vs_independent"] = item["payload_bytes"] / independent_bytes
                item["payload_ratio_vs_source_files"] = item["payload_bytes"] / (source_weights_bytes + source_configs_bytes)
                details["results"].append(item)
                rows.append({
                    "fold": fold, "holdout_task": item["heldout_task"], "k": k,
                    "method": method, "heldout_rel_fro_error": item["heldout_rel_fro_error"],
                    "payload_bytes": item["payload_bytes"],
                    "payload_ratio_vs_independent": item["payload_ratio_vs_independent"],
                    "decode_macs": item["decode_macs"], "fit_wall_seconds": item["fit_wall_seconds"],
                    "reconstruct_wall_seconds": item["reconstruct_wall_seconds"], "gauge_max_error": item["gauge_max_error"],
                    "active_macs_per_token": item["active_macs_per_token"],
                    "status": "MEASURED_ORACLE",
                })
    with (args.out / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (args.out / "results.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
