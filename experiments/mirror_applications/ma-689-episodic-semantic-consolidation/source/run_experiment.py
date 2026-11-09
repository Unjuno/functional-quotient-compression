#!/usr/bin/env python3
"""Deterministic synthetic screen for episodic-state transfer and retirement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = 32
N = 32
N_FIT = 24
RANK = 4
RHOS = (0.0, 0.05, 0.2)
SEEDS = (68901, 68902)


def _payload(**arrays: np.ndarray) -> bytes:
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    return buff.getvalue()


def _write_payload(path: Path, **arrays: np.ndarray) -> int:
    data = _payload(**arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def make_bank(seed: int, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, RANK)))
    shared = rng.standard_normal((N, RANK)) @ q.T
    raw = rng.standard_normal((N, D))
    private = raw - (raw @ q) @ q.T
    shared_norm = np.linalg.norm(shared, axis=1, keepdims=True)
    private_norm = np.linalg.norm(private, axis=1, keepdims=True)
    private = private * (shared_norm / np.maximum(private_norm, 1e-12))
    return (np.sqrt(1.0 - rho * rho) * shared + rho * private).astype(np.float32)


def fit_codes(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # No centering: this is a shared linear basis and coefficients, not an affine model.
    _, _, vt = np.linalg.svd(train.astype(np.float64), full_matrices=False)
    basis = vt[:RANK].astype(np.float32)
    codes = (train @ basis.T).astype(np.float32)
    return basis, codes


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    return float(np.sqrt(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)) / max(denom, 1e-12))


def decode(payload_path: Path, flavor: str, episode_ids: np.ndarray) -> np.ndarray:
    with np.load(payload_path, allow_pickle=False) as state:
        if flavor == "dense":
            ids = state["episode_ids"]
            key = "memory_values" if "memory_values" in state.files else "episode_values"
            vals = state[key]
            lookup = {int(k): v for k, v in zip(ids, vals)}
            return np.stack([lookup[int(k)] for k in episode_ids])
        if flavor == "mirror":
            ids, basis, codes = state["episode_ids"], state["view_basis"], state["view_codes"]
        elif flavor == "native_pca":
            ids, basis, codes = state["episode_ids"], state["view_basis"], state["view_codes"]
        elif flavor == "episodic_mirror":
            ids, basis, codes = state["episode_ids"], state["basis"], state["codes"]
        elif flavor == "episodic_native":
            ids, basis, codes = state["episode_ids"], state["basis"], state["codes"]
        else:
            raise ValueError(flavor)
        lookup = {int(k): c for k, c in zip(ids, codes)}
        code_rows = np.stack([lookup[int(k)] for k in episode_ids])
        return code_rows @ basis


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    payload_manifest: list[dict[str, object]] = []
    for seed in SEEDS:
        for rho in RHOS:
            bank = make_bank(seed, rho)
            train, heldout = bank[:N_FIT], bank[N_FIT:]
            ids = np.arange(N_FIT, dtype=np.int32)
            heldout_ids = np.arange(N_FIT, N, dtype=np.int32)
            basis, codes = fit_codes(train)
            mirror_recon = codes @ basis
            native_recon = codes @ basis
            run_dir = out / f"seed{seed}_rho{rho:g}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Shared initial learned base is included in every complete-state snapshot.
            base_bytes = _write_payload(run_dir / "base_empty.npz", base_state=np.zeros((1, D), np.float32), format_version=np.array([1], np.uint8))
            ids_full = np.arange(N_FIT, dtype=np.int32)
            full_bytes = _write_payload(run_dir / "episodic_independent.npz", episode_ids=ids_full, episode_values=train)
            mirror_episode_bytes = _write_payload(run_dir / "episodic_mirror.npz", episode_ids=ids, basis=basis, codes=codes)
            native_episode_bytes = _write_payload(run_dir / "episodic_native_pca.npz", episode_ids=ids, basis=basis, codes=codes)

            # Dense consolidation materializes all retained episode functions into base state.
            dense_bytes = _write_payload(run_dir / "consolidated_dense.npz", base_state=np.zeros((1, D), np.float32), episode_ids=ids, memory_values=train)
            mirror_bytes = _write_payload(run_dir / "consolidated_mirror.npz", base_state=np.zeros((1, D), np.float32), episode_ids=ids, view_basis=basis, view_codes=codes)
            native_bytes = _write_payload(run_dir / "consolidated_native_pca.npz", base_state=np.zeros((1, D), np.float32), episode_ids=ids, view_basis=basis, view_codes=codes)

            methods = [
                ("independent", full_bytes, train, 0, run_dir / "episodic_independent.npz", "dense"),
                ("mirror_code", mirror_episode_bytes, mirror_recon, RANK * D, run_dir / "episodic_mirror.npz", "episodic_mirror"),
                ("native_pca", native_episode_bytes, native_recon, RANK * D, run_dir / "episodic_native_pca.npz", "episodic_native"),
                ("consolidated_dense", dense_bytes, train, 0, run_dir / "consolidated_dense.npz", "dense"),
                ("consolidated_mirror", mirror_bytes, mirror_recon, RANK * D, run_dir / "consolidated_mirror.npz", "mirror"),
                ("consolidated_native_pca", native_bytes, native_recon, RANK * D, run_dir / "consolidated_native_pca.npz", "native_pca"),
            ]

            # All snapshots use serialized-state replay; stage sizes include every live payload.
            for method, own_bytes, predicted, lookup_macs, path, flavor in methods:
                replay = decode(path, flavor, ids)
                replay_err = nrmse(replay, train)
                quality = nrmse(predicted, train)
                if abs(replay_err - quality) > 1e-7:
                    raise AssertionError((method, replay_err, quality))
                trainable_bytes = own_bytes
                if method in ("independent", "mirror_code", "native_pca"):
                    complete = base_bytes + own_bytes
                    stage = "before_transfer"
                else:
                    complete = own_bytes
                    stage = "after_episode_store_deleted"
                # Explicitly expose peak live state while the transferred base and source store coexist.
                duplicate_peak = (base_bytes + own_bytes + (full_bytes if method == "consolidated_dense" else mirror_episode_bytes if method == "consolidated_mirror" else native_episode_bytes)) if method.startswith("consolidated_") else complete
                transfer_macs = N_FIT * RANK * D if "mirror" in method or "native_pca" in method else 0
                start = time.perf_counter()
                for _ in range(1000):
                    # Payload I/O was already checked above. Time only the in-memory lookup/decode.
                    _ = predicted.copy() if lookup_macs == 0 else codes @ basis
                elapsed = (time.perf_counter() - start) / 1000
                rows.append({
                    "condition": f"rho{rho:g}_{stage}",
                    "world_or_seed": seed,
                    "method": method,
                    "serialized_bytes": complete,
                    "train_tokens_or_examples": N_FIT,
                    "optimizer_updates": 0,
                    "active_compute_proxy": lookup_macs,
                    "wall_time_s": elapsed,
                    "primary_metric": "training_episode_recall_nrmse",
                    "primary_value": quality,
                    "secondary_metric": "heldout_nrmse/transfer_MAC/duplicate_peak_bytes",
                    "secondary_value": f"{nrmse((heldout @ basis.T) @ basis, heldout):.9g}/{transfer_macs}/{duplicate_peak}",
                    "status_note": f"development; serialized_replay_nrmse={replay_err:.9g}; stage={stage}; fresh sealed"
                })
                payload_manifest.append({
                    "seed": seed, "rho": rho, "method": method,
                    "file": str(path.relative_to(out)), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "replay_max_abs_error": float(np.max(np.abs(replay - predicted))) if replay.size else 0.0
                })

            # Check held-out episode diagnostic without fitting its values.
            heldout_pred = (heldout @ basis.T) @ basis
            # Transfer checkpoint retains source store temporarily; this is a live-state sum, not a new serialization.
            for method, base_size, source_size in (("dense", dense_bytes, full_bytes), ("mirror", mirror_bytes, mirror_episode_bytes), ("native_pca", native_bytes, native_episode_bytes)):
                rows.append({
                    "condition": f"rho{rho:g}_after_transfer_store_retained",
                    "world_or_seed": seed,
                    "method": method,
                    "serialized_bytes": base_bytes + base_size + source_size,
                    "train_tokens_or_examples": N_FIT,
                    "optimizer_updates": 0,
                    "active_compute_proxy": 0,
                    "wall_time_s": 0.0,
                    "primary_metric": "training_episode_recall_nrmse",
                    "primary_value": nrmse(train if method == "dense" else mirror_recon, train),
                    "secondary_metric": "heldout_nrmse/transfer_MAC/duplicate_peak_bytes",
                    "secondary_value": f"{nrmse(heldout_pred, heldout):.9g}/{N_FIT*RANK*D if method != 'dense' else 0}/{base_bytes+base_size+source_size}",
                    "status_note": "development; transfer complete, old episodic payload retained; fresh sealed"
                })

            # Store targets and selected decomposition for artifact-level independent replay/audit.
            np.savez_compressed(run_dir / "development_targets.npz", training_targets=train, heldout_targets=heldout, fitted_basis=basis, fitted_codes=codes)

    csv_path = out / "RESULTS_CORE.csv"
    fields = ["condition", "world_or_seed", "method", "serialized_bytes", "train_tokens_or_examples", "optimizer_updates", "active_compute_proxy", "wall_time_s", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "status_note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "payload_manifest.json").write_text(json.dumps(payload_manifest, indent=2) + "\n", encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({"protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(), "development_seeds": list(SEEDS), "fresh_opened": False, "rows": len(rows), "files": len(payload_manifest)}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "source" / "results")
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
