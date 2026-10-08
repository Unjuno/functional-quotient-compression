#!/usr/bin/env python3
"""MA-1141 MuJoCo behavior-cloning and held-out-pair evaluation screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import gymnasium
import numpy as np
import torch
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
MORPHS = PROTOCOL["simulator"]["morphologies"]
SKILLS = PROTOCOL["simulator"]["skills"]
TRAIN_PAIRS = [(m, s) for m in range(4) for s in range(4) if (m + s) % 2 == 0]
AUDIT_PAIRS = [(m, s) for m in range(4) for s in range(4) if (m + s) % 2 == 1]
METHODS = ("native", "no_code", "film", "mirror")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def pair_id(morph: int, skill: int) -> str:
    return f"m{morph}_s{skill}"


def make_env(morph: int):
    xml_path = ROOT / "assets" / f"ant_morph_{morph}.xml"
    return AntEnv(
        xml_file=str(xml_path), frame_skip=1,
        forward_reward_weight=0.0, ctrl_cost_weight=0.0,
        contact_cost_weight=0.0, healthy_reward=0.0,
        terminate_when_unhealthy=True, reset_noise_scale=0.05,
        exclude_current_positions_from_observation=True,
        include_cfrc_ext_in_observation=True,
    )


PHASE_OFFSETS = (
    (0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
    (0.0, math.pi, math.pi, 0.0),
    (0.0, 0.0, math.pi, math.pi),
    (0.0, math.pi, 0.0, math.pi),
)


def teacher_action(morph: int, skill: int, phase: float) -> np.ndarray:
    target = float(SKILLS[skill]["target_forward_velocity"])
    scale = float(MORPHS[morph]["actuator_scale"])
    amplitude = (0.45 + 0.4 * target) * scale
    offsets = PHASE_OFFSETS[skill]
    # Ant actuator order is leg4, leg1, leg2, leg3; semantic legs are FL, FR, BL, BR.
    semantic = []
    for offset in offsets:
        phi = phase + offset
        semantic.extend((amplitude * math.sin(phi), amplitude * math.cos(phi)))
    # semantic index pairs: FL(hip1,ankle1), FR(hip2,ankle2), BL(hip3,ankle3), BR(hip4,ankle4)
    return np.asarray([semantic[6], semantic[7], semantic[0], semantic[1], semantic[2], semantic[3], semantic[4], semantic[5]], dtype=np.float32).clip(-1, 1)


def input_features(obs: np.ndarray, phase: float) -> np.ndarray:
    return np.concatenate((obs.astype(np.float32), np.asarray([math.sin(phase), math.cos(phase)], dtype=np.float32)))


def condition_features(base: np.ndarray, morph: np.ndarray, skill: np.ndarray) -> np.ndarray:
    one_m = np.zeros(4, dtype=np.float32); one_m[morph] = 1
    one_s = np.zeros(4, dtype=np.float32); one_s[skill] = 1
    return np.concatenate((base, one_m, one_s))


class Policy(torch.nn.Module):
    def __init__(self, method: str, base_dim: int):
        super().__init__()
        self.method = method
        in_dim = base_dim + (8 if method == "native" else 0)
        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, 8)
        if method in ("film", "mirror"):
            self.morph_codes = torch.nn.Parameter(torch.zeros(4, 2))
            self.skill_codes = torch.nn.Parameter(torch.zeros(4, 2))
        if method == "film":
            self.gamma_basis = torch.nn.Parameter(torch.randn(2, 128) * .02)
            self.beta_basis = torch.nn.Parameter(torch.randn(2, 128) * .02)

    def forward(self, x, morph=None, skill=None):
        if self.method == "native":
            one_m = torch.nn.functional.one_hot(morph, 4).to(x.dtype)
            one_s = torch.nn.functional.one_hot(skill, 4).to(x.dtype)
            x = torch.cat((x, one_m, one_s), dim=-1)
        h = torch.relu(self.fc1(x))
        if self.method == "film":
            z = self.morph_codes[morph] + self.skill_codes[skill]
            gamma = z @ self.gamma_basis
            beta = z @ self.beta_basis
            h = h * (1 + gamma) + beta
        elif self.method == "mirror":
            for code in (self.morph_codes[morph], self.skill_codes[skill]):
                parts = list(torch.unbind(h, dim=1))
                for j, (a, b) in enumerate(((0, 1), (2, 3))):
                    theta = code[:, j]
                    co, si = torch.cos(theta), torch.sin(theta)
                    ha, hb = parts[a], parts[b]
                    parts[a] = co * ha - si * hb
                    parts[b] = si * ha + co * hb
                h = torch.stack(parts, dim=1)
        h = torch.relu(self.fc2(h))
        return torch.tanh(self.out(h))


def collect_episode(morph: int, skill: int, seed: int, horizon: int):
    env = make_env(morph)
    obs, _ = env.reset(seed=seed)
    target = float(SKILLS[skill]["target_forward_velocity"])
    phase = 0.0
    dt = float(PROTOCOL["simulator"]["control_timestep_seconds"])
    frequency = 1.0 + 1.5 * target
    xs, ys = [], []
    for _ in range(horizon):
        x = input_features(obs, phase)
        action = teacher_action(morph, skill, phase)
        xs.append(x); ys.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        phase += 2 * math.pi * frequency * dt
        if terminated or truncated:
            break
    env.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect_world(world_seed: int, horizon: int = 1000):
    train_x, train_y, train_m, train_s = [], [], [], []
    dev_x, dev_y, dev_m, dev_s = [], [], [], []
    pair_rows = []
    for mi, si in TRAIN_PAIRS:
        for ep in range(5):
            seed = world_seed * 100000 + (mi * 4 + si) * 100 + ep
            x, y = collect_episode(mi, si, seed, horizon)
            train_x.append(x); train_y.append(y)
            train_m.extend([mi] * len(x)); train_s.extend([si] * len(x))
            pair_rows.append({"split": "train", "pair": pair_id(mi, si), "seed": seed, "examples": len(x)})
        for seed0 in PROTOCOL["simulator"]["episode_seeds"]["development"]:
            seed = world_seed * 100000 + (mi * 4 + si) * 100 + seed0
            x, y = collect_episode(mi, si, seed, horizon)
            dev_x.append(x); dev_y.append(y)
            dev_m.extend([mi] * len(x)); dev_s.extend([si] * len(x))
            pair_rows.append({"split": "development", "pair": pair_id(mi, si), "seed": seed, "examples": len(x)})
    def pack(xs, ys, ms, ss):
        return (np.concatenate(xs), np.concatenate(ys), np.asarray(ms, np.int64), np.asarray(ss, np.int64))
    return pack(train_x, train_y, train_m, train_s), pack(dev_x, dev_y, dev_m, dev_s), pair_rows


def batch_inputs(model: Policy, x, morph, skill):
    xt = torch.as_tensor(x, dtype=torch.float32)
    mt = torch.as_tensor(morph, dtype=torch.long)
    st = torch.as_tensor(skill, dtype=torch.long)
    return model(xt, mt, st)


def fit(model: Policy, train_data, dev_data, max_updates=1500):
    tx, ty, tm, ts = train_data
    dx, dy, dm, ds = dev_data
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best = math.inf; best_state = None; stale = 0; n = len(tx)
    rng = np.random.default_rng(817)
    t0 = time.perf_counter()
    updates = 0
    while updates < max_updates:
        idx = rng.integers(0, n, size=256)
        pred = batch_inputs(model, tx[idx], tm[idx], ts[idx])
        target = torch.as_tensor(ty[idx])
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        updates += 1
        if updates % 20 == 0:
            with torch.no_grad():
                pred = batch_inputs(model, dx, dm, ds)
                score = torch.nn.functional.mse_loss(pred, torch.as_tensor(dy)).item()
            if score < best - 1e-8:
                best = score; stale = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 150:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return updates, len(tx), best, time.perf_counter() - t0


def episode_rollout(model, method, morph, skill, seed, horizon=1000, oracle=False):
    env = make_env(morph)
    obs, _ = env.reset(seed=seed)
    phase = 0.0
    target = float(SKILLS[skill]["target_forward_velocity"])
    freq = 1.0 + 1.5 * target
    dt = float(PROTOCOL["simulator"]["control_timestep_seconds"])
    total_return = speed_err = fall_steps = action_cost = 0.0
    rollout_start = time.perf_counter()
    fallen = False
    for step in range(horizon):
        if oracle:
            action = teacher_action(morph, skill, phase)
        else:
            feat = input_features(obs, phase)[None, :]
            with torch.no_grad():
                action = model(torch.as_tensor(feat), torch.tensor([morph]), torch.tensor([skill]))[0].numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        vx, vy = float(env.data.qvel[0]), float(env.data.qvel[1])
        z = float(env.data.qpos[2])
        unhealthy = terminated or truncated or z < .2 or z > 1.0
        if unhealthy:
            fallen = True
        if fallen:
            r = -1.0; fall_steps += 1
        else:
            r = math.exp(-abs(vx - target) / max(target, .15)) - .1 * abs(vy) - .001 * float(np.square(action).sum())
        total_return += r
        speed_err += abs(vx - target)
        action_cost += float(np.square(action).sum())
        phase += 2 * math.pi * freq * dt
        if fallen:
            # Count the remaining horizon as fall penalty without running a fallen simulation.
            remain = horizon - step - 1
            total_return -= float(remain)
            fall_steps += remain
            speed_err += remain * max(target, .15)
            action_cost += remain * 8
            break
    elapsed = time.perf_counter() - rollout_start
    env.close()
    return {"mean_return": total_return / horizon, "mean_target_speed_error": speed_err / horizon, "fall_fraction": fall_steps / horizon, "mean_action_sq": action_cost / horizon, "rollout_wall_seconds": elapsed}


def model_state_bytes(model, method, stem):
    ART.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    code_keys = [k for k in state if k in ("morph_codes", "skill_codes")]
    server = {k: v for k, v in state.items() if k not in code_keys}
    code = {k: state[k] for k in code_keys}
    meta = {"method": method, "morphology_ids": list(range(4)), "skill_ids": list(range(4)), "mapping": "source/protocol.json"}
    sp = ART / f"{stem}_server.safetensors"
    save_file(server, str(sp), metadata={"experiment": "MA-1141", "kind": "shared_policy"})
    cp = None
    if code:
        cp = ART / f"{stem}_codes.safetensors"
        save_file(code, str(cp), metadata={"experiment": "MA-1141", "kind": "factor_codes"})
    mp = ART / f"{stem}_metadata.json"
    mp.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n")
    server_bytes = sp.stat().st_size + mp.stat().st_size
    code_bytes = cp.stat().st_size if cp else 0
    per_pair_bytes = 0
    if code:
        pair_tensors = {"morph": state["morph_codes"][0], "skill": state["skill_codes"][0]}
        pp = ART / f"{stem}_pair.safetensors"
        save_file(pair_tensors, str(pp), metadata={"experiment": "MA-1141", "kind": "pair_code"})
        per_pair_bytes = pp.stat().st_size
    return server_bytes, code_bytes, per_pair_bytes, server_bytes + code_bytes


def policy_macs(method, base_dim):
    n = base_dim + (8 if method == "native" else 0)
    mac = n * 128 + 128 * 128 + 128 * 8
    if method == "film": mac += 2 * 128 * 2 + 2 * 128
    if method == "mirror": mac += 32
    return int(mac)


def measure_latency(model, method, base_dim):
    x = torch.zeros(1, base_dim)
    m = torch.tensor([0]); s = torch.tensor([0])
    for _ in range(50): model(x, m, s)
    t0 = time.perf_counter()
    for _ in range(500): model(x, m, s)
    return (time.perf_counter() - t0) * 1000 / 500


def train_evaluate(world_seeds=None, model_seeds=None, max_updates=1500, audit=False):
    ART.mkdir(parents=True, exist_ok=True)
    world_seeds = world_seeds or PROTOCOL["training"]["fresh_world_seeds"]
    model_seeds = model_seeds or PROTOCOL["training"]["model_init_seeds"]
    rows = []
    for world in world_seeds:
        train_data, dev_data, pair_rows = collect_world(world, PROTOCOL["simulator"]["rollout_horizon_steps"])
        xtr, ytr, mtr, str_ = train_data
        xdv, ydv, mdv, sdv = dev_data
        # Persist exact training/dev split provenance and content hashes, not a simulator checkpoint.
        split_path = ART / f"world_{world}_split_manifest.json"
        split_obj = {"world_seed": world, "train_examples": len(xtr), "dev_examples": len(xdv), "train_pairs": [pair_id(*p) for p in TRAIN_PAIRS], "dev_pairs": [pair_id(*p) for p in TRAIN_PAIRS], "audit_pairs": [pair_id(*p) for p in AUDIT_PAIRS], "episodes": pair_rows}
        split_path.write_text(json.dumps(split_obj, indent=2) + "\n")
        np.savez_compressed(ART / f"world_{world}_demo_data.npz", train_x=xtr, train_y=ytr, train_m=mtr, train_s=str_, dev_x=xdv, dev_y=ydv, dev_m=mdv, dev_s=sdv)
        base_dim = xtr.shape[1]
        for init_seed in model_seeds:
            for method in METHODS:
                set_seed(init_seed)
                model = Policy(method, base_dim)
                updates, ntrain, dev_mse, train_wall = fit(model, train_data, dev_data, max_updates)
                stem = f"w{world}_s{init_seed}_{method}"
                server_b, code_b, pair_b, total_b = model_state_bytes(model, method, stem)
                latency = measure_latency(model, method, base_dim)
                # Audit runs occur only after all selected checkpoints are fixed from seen-pair dev data.
                eval_pairs = AUDIT_PAIRS if audit else TRAIN_PAIRS
                eval_seeds = PROTOCOL["simulator"]["episode_seeds"]["audit"] if audit else PROTOCOL["simulator"]["episode_seeds"]["development"]
                for mi, si in eval_pairs:
                    for seed0 in eval_seeds:
                        seed = world * 100000 + (mi * 4 + si) * 100 + seed0
                        metrics = episode_rollout(model, method, mi, si, seed, PROTOCOL["simulator"]["rollout_horizon_steps"])
                        row = {"world_seed": world, "model_seed": init_seed, "method": method, "pair_id": pair_id(mi, si), "morphology_id": mi, "skill_id": si, "episode_seed": seed, "updates": updates, "train_examples": ntrain, "action_mse": dev_mse, **metrics, "server_payload_bytes": server_b, "registered_factor_code_bytes": code_b, "per_pair_download_bytes": pair_b, "full_policy_bytes": total_b, "action_macs": policy_macs(method, base_dim), "cpu_batch1_latency_ms": latency, "train_wall_seconds": train_wall, "status": "audit" if audit else "development"}
                        rows.append(row)
                # Teacher score is logged separately and never used for checkpoint selection.
                teacher_pairs = eval_pairs
                for mi, si in teacher_pairs:
                    for seed0 in eval_seeds:
                        seed = world * 100000 + (mi * 4 + si) * 100 + seed0
                        metrics = episode_rollout(model, method, mi, si, seed, PROTOCOL["simulator"]["rollout_horizon_steps"], oracle=True)
                        rows.append({"world_seed": world, "model_seed": init_seed, "method": "teacher", "pair_id": pair_id(mi, si), "morphology_id": mi, "skill_id": si, "episode_seed": seed, "updates": 0, "train_examples": 0, "action_mse": 0.0, **metrics, "server_payload_bytes": 0, "registered_factor_code_bytes": 0, "per_pair_download_bytes": 0, "full_policy_bytes": 0, "action_macs": 0, "cpu_batch1_latency_ms": 0, "train_wall_seconds": 0, "status": "oracle_reference"})
    out = ART / ("metrics_audit.csv" if audit else "metrics_development.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-seeds", nargs="+", type=int)
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--max-updates", type=int, default=1500)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    print(train_evaluate(args.world_seeds, args.model_seeds, args.max_updates, args.audit))
