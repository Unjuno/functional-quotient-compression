#!/usr/bin/env python3
"""Small synthetic scene-mode CP attribution screen; no real rendering claims."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "RESULTS_CORE.csv"
N = 8
RANK = 4
N_DEV = 6
N_FRESH = 3
LR = 0.03
CP_STEPS = 1000
CODE_STEPS = 500
OBSERVED_FRACTION = 0.5


def make_truth(seed: int):
    """Return six development and three fresh scalar fields with shared CP atoms."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.linspace(-1.0, 1.0, N)
    factors = []
    for axis in range(3):
        # Smooth but distinct separable atoms. Random signed phases vary by seed.
        cols = []
        for k in range(RANK):
            phase = (torch.rand((), generator=gen).item() - 0.5) * 0.25
            col = torch.sin(math.pi * (k + 1) * (coords + phase))
            col = col / col.square().mean().sqrt()
            cols.append(col)
        factors.append(torch.stack(cols, dim=1))
    codes = torch.randn((N_DEV + N_FRESH, RANK), generator=gen) * 0.075
    fields = torch.einsum("xr,yr,zr,sr->sxyz", *factors, codes) + 0.5
    return fields[:N_DEV], fields[N_DEV:], *factors, codes


def cp_eval(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    return torch.einsum("xr,yr,zr,sr->sxyz", x, y, z, code) + 0.5


def bytes_npz(arrays: dict[str, np.ndarray], metadata: dict) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays, metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8))
    return bio.getvalue()


def fit_shared(train: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed + 101)
    x = torch.nn.Parameter(torch.randn(N, RANK) * 0.65)
    y = torch.nn.Parameter(torch.randn(N, RANK) * 0.65)
    z = torch.nn.Parameter(torch.randn(N, RANK) * 0.65)
    code = torch.nn.Parameter(torch.randn(N_DEV, RANK) * 0.04)
    opt = torch.optim.Adam([x, y, z, code], lr=LR)
    started = time.perf_counter()
    for _ in range(CP_STEPS):
        opt.zero_grad(set_to_none=True)
        pred = cp_eval(x, y, z, code)
        loss = (pred - train).square().mean()
        loss.backward()
        opt.step()
    return x.detach(), y.detach(), z.detach(), code.detach(), time.perf_counter() - started


def fit_fresh_codes(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, truth: torch.Tensor, seed: int):
    torch.manual_seed(seed + 202)
    code = torch.nn.Parameter(torch.randn(N_FRESH, RANK) * 0.03)
    opt = torch.optim.Adam([code], lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(seed + 303)
    obs, held = coordinate_split(seed)
    coords = torch.stack(torch.meshgrid(*(torch.arange(N) for _ in range(3)), indexing="ij"), dim=-1).reshape(-1, 3).float()
    # Fit codes against observed points only, with the shared spatial factors frozen.
    xq, yq, zq = x[coords[:, 0].long()], y[coords[:, 1].long()], z[coords[:, 2].long()]
    basis = xq[:, :, None] * yq[:, None, :]  # construct below as rank-wise products
    basis = xq * yq * zq
    started = time.perf_counter()
    for _ in range(CODE_STEPS):
        opt.zero_grad(set_to_none=True)
        pred = code @ basis[obs].T + 0.5
        loss = (pred - truth.reshape(N_FRESH, -1)[:, obs]).square().mean()
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - started
    pred = code.detach() @ basis[held].T + 0.5
    actual = truth.reshape(N_FRESH, -1)[:, held]
    rmse = float((pred - actual).square().mean().sqrt())
    return code.detach(), rmse, elapsed, obs, held


def coordinate_split(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed + 303)
    perm = torch.randperm(N**3, generator=gen)
    nobs = int(N**3 * OBSERVED_FRACTION)
    return perm[:nobs], perm[nobs:]


def psnr_proxy(rmse: float) -> float:
    return float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))


def run(seed: int, phase: str) -> list[dict]:
    train, fresh, *_ = make_truth(seed)
    x, y, z, dev_code, fit_s = fit_shared(train, seed)
    fresh_code, _, code_s, obs, held = fit_fresh_codes(x, y, z, fresh, seed)
    pred_cp = cp_eval(x, y, z, fresh_code).reshape(N_FRESH, -1)[:, held]
    truth_held = fresh.reshape(N_FRESH, -1)[:, held]
    cp_rmse = float((pred_cp - truth_held).square().mean().sqrt())

    # Native 4-way CP and Mirror scene-code CP are the same trained arrays/function.
    cp_arrays = {"x": x.numpy(), "y": y.numpy(), "z": z.numpy(), "scene_code": fresh_code.numpy()}
    cp_payload = bytes_npz(cp_arrays, {"grid": [N, N, N], "rank": RANK, "offset": 0.5})

    # PCA over the flattened development fields; fresh codes use the same observed voxels.
    flat = train.reshape(N_DEV, -1)
    mean = flat.mean(0)
    _, _, vh = torch.linalg.svd(flat - mean, full_matrices=False)
    basis_pca = vh[:RANK].T.contiguous()
    fresh_center = fresh.reshape(N_FRESH, -1) - mean
    pcode = torch.nn.Parameter(torch.zeros(N_FRESH, RANK))
    popt = torch.optim.Adam([pcode], lr=LR)
    started = time.perf_counter()
    for _ in range(CODE_STEPS):
        popt.zero_grad(set_to_none=True)
        pred = pcode @ basis_pca[obs].T + mean[obs]
        loss = (pred - fresh.reshape(N_FRESH, -1)[:, obs]).square().mean()
        loss.backward()
        popt.step()
    pca_code_s = time.perf_counter() - started
    pca_pred = pcode.detach() @ basis_pca[held].T + mean[held]
    pca_rmse = float((pca_pred - truth_held).square().mean().sqrt())
    pca_payload = bytes_npz({"mean": mean.numpy(), "basis": basis_pca.numpy(), "scene_code": pcode.detach().numpy()}, {"grid": [N, N, N], "rank": RANK})

    dense = fresh.numpy().astype(np.float32)
    dense_payload = bytes_npz({"fields": dense}, {"grid": [N, N, N], "scene_count": N_FRESH})
    dense_rmse = 0.0  # Independent upper control stores the exact fresh fields.

    rows = []
    for method, rmse, payload, fit_time, code_time, macs in [
        ("mirror_shared_cp", cp_rmse, cp_payload, fit_s, code_s, (4 * RANK - 1) * len(held) * N_FRESH),
        ("native_scene_mode_cp", cp_rmse, cp_payload, fit_s, code_s, (4 * RANK - 1) * len(held) * N_FRESH),
        ("pca_flat_fields", pca_rmse, pca_payload, 0.0, pca_code_s, 2 * RANK * len(held) * N_FRESH),
        ("independent_dense", dense_rmse, dense_payload, 0.0, 0.0, len(held) * N_FRESH),
    ]:
        # Time the held-coordinate predictions on CPU, repeat enough to exceed timer granularity.
        start = time.perf_counter()
        for _ in range(100):
            if method in ("mirror_shared_cp", "native_scene_mode_cp"):
                _ = cp_eval(x, y, z, fresh_code)
            elif method == "pca_flat_fields":
                _ = pcode.detach() @ basis_pca.T + mean
            else:
                _ = torch.from_numpy(dense).reshape(N_FRESH, -1)
        query_s = (time.perf_counter() - start) / 100
        rows.append({
            "method": method,
            "phase": phase,
            "seed": seed,
            "development_scenes": N_DEV,
            "fresh_scenes": N_FRESH,
            "observed_fraction": OBSERVED_FRACTION,
            "heldout_nrmse": rmse,
            "field_psnr_proxy": psnr_proxy(rmse),
            "payload_bytes": len(payload),
            "fit_seconds": fit_time,
            "code_fit_seconds": code_time,
            "query_seconds": query_s,
            "mac_proxy": macs,
            "notes": "synthetic scalar lattice; not rendered views",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["development", "fresh"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    rows = [row for seed in args.seeds for row in run(seed, args.phase)]
    mode = "w" if args.phase == "development" else "a"
    with RESULTS.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if mode == "w": writer.writeheader()
        writer.writerows(rows)
    summary_path = ROOT / "source" / f"screen_summary_{args.phase}.json"
    summary_path.write_text(json.dumps({"phase": args.phase, "seeds": args.seeds, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "seeds": args.seeds, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
