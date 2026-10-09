#!/usr/bin/env python3
"""MA-503 synthetic factorized layer x task representation intervention screen."""
import argparse
import csv
import hashlib
import io
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PAYLOADS = ART / "payloads"
D, R, L, T = 64, 4, 8, 32
DEV_WORLDS = [50300, 50301]
FRESH_WORLDS = [50320, 50321, 50322]
SEEDS = [0, 1, 2]
STEP_OPTIONS = [400, 800, 1200]
METHODS = ["dense_independent", "loreft_pair_table", "shared_task_tie", "diag_layer_factor", "generic_full_layer", "mirror_givens"]
torch.set_num_threads(1)


def make_world(world: int, seed: int, rho: float):
    gen = torch.Generator().manual_seed(world * 100003 + seed * 7919 + int(rho * 1000) * 101 + 53)
    basis = torch.linalg.qr(torch.randn(D, R, generator=gen)).Q.contiguous()
    z = torch.randn(T, R, generator=gen) * 0.6
    angles = (torch.rand(L, 2, generator=gen) - 0.5) * 2.0
    coeff = torch.empty(L, T, R)
    for layer in range(L):
        for task in range(T):
            v = z[task].clone()
            a, b = angles[layer]
            c, s = torch.cos(a), torch.sin(a)
            v[0], v[1] = c * z[task, 0] - s * z[task, 1], s * z[task, 0] + c * z[task, 1]
            c, s = torch.cos(b), torch.sin(b)
            v[2], v[3] = c * z[task, 2] - s * z[task, 3], s * z[task, 2] + c * z[task, 3]
            coeff[layer, task] = v
    if rho:
        coeff += rho * torch.randn(L, T, R, generator=gen)
    targets = coeff @ basis.T
    # Each layer and task is represented in training. Holdout mask is deterministic by ID.
    held = torch.tensor([[(layer * 7 + task * 11 + seed) % 5 == 0 for task in range(T)] for layer in range(L)])
    for layer in range(L):
        if held[layer].all():
            held[layer, 0] = False
    for task in range(T):
        if held[:, task].all():
            held[0, task] = False
    return basis, coeff, targets, held


def givens_view(z, angle):
    a = angle[:, 0]
    b = angle[:, 1]
    out = z.unsqueeze(0).expand(len(a), -1, -1).clone()
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    x, y = out[:, :, 0].clone(), out[:, :, 1].clone()
    out[:, :, 0] = c * x - s * y
    out[:, :, 1] = s * x + c * y
    c, s = torch.cos(b)[:, None], torch.sin(b)[:, None]
    x, y = out[:, :, 2].clone(), out[:, :, 3].clone()
    out[:, :, 2] = c * x - s * y
    out[:, :, 3] = s * x + c * y
    return out


def fit_model(method, basis, coeff, held, steps, seed):
    train = ~held
    gen = torch.Generator().manual_seed(seed + 777)
    if method == "mirror_givens":
        z = torch.nn.Parameter(torch.randn(T, R, generator=gen) * 0.1)
        angle = torch.nn.Parameter(torch.zeros(L, 2))
        params = [z, angle]
        def predict(): return givens_view(z, angle)
    elif method == "generic_full_layer":
        q = torch.nn.Parameter(torch.randn(T, R, generator=gen) * 0.1)
        A = torch.nn.Parameter(torch.eye(R).unsqueeze(0).repeat(L, 1, 1))
        params = [q, A]
        def predict(): return torch.einsum("lij,tj->lti", A, q)
    elif method == "diag_layer_factor":
        q = torch.nn.Parameter(torch.randn(T, R, generator=gen) * 0.1)
        a = torch.nn.Parameter(torch.ones(L, R))
        params = [q, a]
        def predict(): return a[:, None, :] * q[None, :, :]
    elif method == "shared_task_tie":
        q = torch.nn.Parameter(torch.randn(T, R, generator=gen) * 0.1)
        params = [q]
        def predict(): return q.unsqueeze(0).expand(L, -1, -1)
    else:
        raise ValueError(method)
    opt = torch.optim.Adam(params, lr=0.025)
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = predict()
        loss = ((pred[train] - coeff[train]) ** 2).mean()
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        state = {"method": method, "basis": basis.detach().clone()}
        if method == "mirror_givens": state.update(z=z.detach().clone(), angles=angle.detach().clone())
        elif method == "generic_full_layer": state.update(task_codes=q.detach().clone(), layer_matrices=A.detach().clone())
        elif method == "diag_layer_factor": state.update(task_codes=q.detach().clone(), layer_scale=a.detach().clone())
        else: state.update(task_codes=q.detach().clone())
    return state, elapsed


def predict_coeff(state):
    method = state["method"]
    if method == "mirror_givens": return givens_view(state["z"], state["angles"])
    if method == "generic_full_layer": return torch.einsum("lij,tj->lti", state["layer_matrices"], state["task_codes"])
    if method == "diag_layer_factor": return state["layer_scale"][:, None, :] * state["task_codes"][None, :, :]
    if method == "shared_task_tie": return state["task_codes"].unsqueeze(0).expand(L, -1, -1)
    if method == "loreft_pair_table": return state["coefficients"]
    if method == "dense_independent": return torch.zeros(L, T, R)
    raise ValueError(method)


def build_state(method, basis, coeff, held, steps, seed):
    if method in {"mirror_givens", "generic_full_layer", "diag_layer_factor", "shared_task_tie"}:
        return fit_model(method, basis, coeff, held, steps, seed)
    if method == "loreft_pair_table":
        return {"method": method, "basis": basis, "coefficients": coeff.clone()}, 0.0
    if method == "dense_independent":
        return {"method": method, "targets": coeff @ basis.T}, 0.0
    raise ValueError(method)


def serialize_state(state):
    b = io.BytesIO()
    torch.save(state, b)
    return b.getvalue()


def evaluate(method, state, basis, coeff, targets, held, steps, fit_s, world, seed, rho, phase):
    payload = serialize_state(state)
    pred_c = predict_coeff(state)
    if method == "dense_independent":
        pred_y = state["targets"]
    else:
        pred_y = pred_c @ state["basis"].T
    eps = 1e-12
    def nrmse(mask):
        delta = pred_y[mask] - targets[mask]
        ref = targets[mask]
        return float(torch.linalg.norm(delta) / (torch.linalg.norm(ref) + eps))
    # Decode latency is measured over a complete layer x task batch after warmup.
    for _ in range(3):
        _ = predict_coeff(state) @ basis.T
    ts = time.perf_counter()
    for _ in range(30):
        _ = predict_coeff(state) @ basis.T
    decode_s = (time.perf_counter() - ts) / 30
    digest = hashlib.sha256(payload).hexdigest()
    mac = L * T * (D * R + (2 * R if method == "mirror_givens" else R * R if method == "generic_full_layer" else 2 * R if method == "diag_layer_factor" else R))
    path = PAYLOADS / f"{phase}_{world}_{seed}_rho{rho:.1f}_{method}.pt"
    path.write_bytes(payload)
    # Verify that predictions reconstructed from the serialized payload are deterministic.
    reloaded = torch.load(io.BytesIO(payload), weights_only=False)
    rec_c = predict_coeff(reloaded)
    rec_y = reloaded["targets"] if method == "dense_independent" else rec_c @ reloaded["basis"].T
    max_replay = float((rec_y - pred_y).abs().max())
    return {
        "phase": phase, "world": world, "seed": seed, "rho": rho, "method": method,
        "steps": steps, "fit_wall_seconds": fit_s, "decode_wall_seconds_batch": decode_s,
        "heldout_nrmse": nrmse(held), "observed_nrmse": nrmse(~held),
        "serialized_payload_bytes": len(payload), "bytes_per_layer_task_pair": len(payload) / (L*T),
        "application_mac_proxy_per_pair": mac / (L*T), "payload_sha256": digest,
        "max_serialized_replay_abs_error": max_replay,
        "payload_path": str(path.relative_to(ROOT.parents[2]))
    }


def run_development():
    ART.mkdir(exist_ok=True); PAYLOADS.mkdir(exist_ok=True)
    rows = []
    for steps in STEP_OPTIONS:
        for world in DEV_WORLDS:
            for seed in SEEDS:
                basis, coeff, targets, held = make_world(world, seed, 0.0)
                for method in ["mirror_givens", "generic_full_layer"]:
                    state, fit_s = build_state(method, basis, coeff, held, steps, world * 100 + seed)
                    rows.append(evaluate(method, state, basis, coeff, targets, held, steps, fit_s, world, seed, 0.0, "development"))
    with (ROOT / "artifacts" / "development_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    score = {}
    for steps in STEP_OPTIONS:
        vals = [r["heldout_nrmse"] for r in rows if r["steps"] == steps and r["method"] == "mirror_givens"]
        score[steps] = statistics.mean(vals)
    selected = min(STEP_OPTIONS, key=lambda s: score[s])
    (ART / "selected_steps.json").write_text(json.dumps({"selected_steps": selected, "dev_mean_mirror_heldout_nrmse": score}, indent=2) + "\n")
    print(json.dumps({"phase": "development", "selected_steps": selected, "scores": score}, indent=2))


def run_fresh():
    lock = json.loads((ART / "selected_steps.json").read_text())
    steps = int(lock["selected_steps"])
    rows = []
    for rho in [0.0, 0.1]:
        for world in FRESH_WORLDS:
            for seed in SEEDS:
                basis, coeff, targets, held = make_world(world, seed, rho)
                for method in METHODS:
                    state, fit_s = build_state(method, basis, coeff, held, steps, world * 100 + seed)
                    rows.append(evaluate(method, state, basis, coeff, targets, held, steps, fit_s, world, seed, rho, "fresh"))
    with (ROOT / "RESULTS_CORE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (ART / "fresh_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"phase": "fresh", "rows": len(rows), "steps": steps}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "fresh"], required=True)
    args = parser.parse_args()
    if args.phase == "development": run_development()
    else: run_fresh()

if __name__ == "__main__": main()
