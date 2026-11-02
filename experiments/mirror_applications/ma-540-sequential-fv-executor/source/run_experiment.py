"""Frozen MA-540 sequential function-vector executor screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
K, STATES, WIDTH, HIDDEN = 8, 16, 16, 48
HELDOUT = ((0, 1), (1, 0), (2, 3), (3, 2))
METHODS = ("fv_tied", "native_tied", "fv_one_shot", "native_untied")
DEV_WORLDS = (54001, 54002)
FRESH_WORLDS = (54011, 54012, 54013)


def bits(x: np.ndarray) -> np.ndarray:
    return ((x[..., None] >> np.arange(4, dtype=np.int64)) & 1).astype(np.uint8)


def random_invertible(rng: np.random.Generator) -> np.ndarray:
    while True:
        a = rng.integers(0, 2, size=(4, 4), dtype=np.uint8)
        # GF(2) Gaussian elimination.
        work = a.copy()
        rank = 0
        for col in range(4):
            pivots = np.flatnonzero(work[rank:, col])
            if len(pivots):
                row = rank + int(pivots[0])
                work[[rank, row]] = work[[row, rank]]
                for other in range(4):
                    if other != rank and work[other, col]:
                        work[other] ^= work[rank]
                rank += 1
        if rank == 4:
            return a


def affine_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.arange(STATES, dtype=np.int64)
    yb = (bits(x) @ a.T + b) % 2
    return (yb.astype(np.int64) * (1 << np.arange(4))).sum(-1).astype(np.int64)


def make_world(seed: int):
    rng = np.random.default_rng(seed)
    maps = []
    signatures = set()
    while len(maps) < K:
        table = affine_table(random_invertible(rng), rng.integers(0, 2, size=4, dtype=np.uint8))
        signature = tuple(table.tolist())
        if signature not in signatures:
            signatures.add(signature)
            maps.append(table)
    maps = np.stack(maps)
    support = np.empty((K, 8, 2), dtype=np.int64)
    for op in range(K):
        xs = rng.permutation(STATES)[:8]
        support[op, :, 0] = xs
        support[op, :, 1] = maps[op, xs]
    atoms_x, atoms_op, atoms_y = [], [], []
    pair_x, pair_ops, pair_y, pair_mid = [], [], [], []
    for op in range(K):
        for x in range(STATES):
            atoms_x.append(x); atoms_op.append(op); atoms_y.append(maps[op, x])
    for first in range(K):
        for second in range(K):
            if (first, second) in HELDOUT:
                continue
            for x in range(STATES):
                mid = maps[first, x]
                pair_x.append(x); pair_ops.append((first, second)); pair_mid.append(mid)
                pair_y.append(maps[second, mid])
    held = []
    for first, second in HELDOUT:
        for x in range(STATES):
            mid = maps[first, x]
            held.append((first, second, x, mid, int(maps[second, mid])))
    return maps, support, (np.array(atoms_op), np.array(atoms_x), np.array(atoms_y)), (
        np.array(pair_ops), np.array(pair_x), np.array(pair_mid), np.array(pair_y)
    ), np.array(held, dtype=np.int64)


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Embedding(STATES, WIDTH)
        self.phase = nn.Parameter(torch.zeros(2, WIDTH))
        self.layers = nn.Sequential(
            nn.Linear(2 * WIDTH, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, STATES),
        )

    def forward(self, state_distribution: torch.Tensor, code: torch.Tensor, step: int):
        state_vec = state_distribution @ self.state.weight
        inp = torch.cat((state_vec, code + self.phase[step]), dim=-1)
        return self.layers(inp)


class Executor(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        self.method = method
        self.state = nn.Embedding(STATES, WIDTH)
        if method.startswith("fv_"):
            self.target = nn.Embedding(STATES, WIDTH)
            self.pair_encoder = nn.Linear(2 * WIDTH, WIDTH)
        else:
            self.native_codes = nn.Parameter(torch.randn(K, WIDTH) * 0.02)
        self.phase = nn.Parameter(torch.zeros(2, WIDTH))
        if method == "native_untied":
            self.transitions = nn.ModuleList([self._make_transition(), self._make_transition()])
        elif method == "fv_one_shot":
            self.one_shot = nn.Sequential(
                nn.Linear(3 * WIDTH, HIDDEN), nn.GELU(),
                nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                nn.Linear(HIDDEN, STATES),
            )
        else:
            self.transition = self._make_transition()

    @staticmethod
    def _make_transition():
        return nn.Sequential(
            nn.Linear(2 * WIDTH, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, STATES),
        )

    def function_codes(self, support: torch.Tensor):
        if self.method.startswith("fv_"):
            src = self.state(support[..., 0].long())
            dst = self.target(support[..., 1].long())
            return torch.tanh(self.pair_encoder(torch.cat((src, dst), dim=-1))).mean(dim=1)
        return self.native_codes

    def _step(self, distribution: torch.Tensor, code: torch.Tensor, step: int):
        state_vec = distribution @ self.state.weight
        inp = torch.cat((state_vec, code + self.phase[step]), dim=-1)
        block = self.transitions[step] if self.method == "native_untied" else self.transition
        return block(inp)

    def forward(self, ops: torch.Tensor, states: torch.Tensor, support: torch.Tensor,
                hard_feedback: bool = False):
        codes = self.function_codes(support)
        ops = ops.long()
        states = states.long()
        onehot = F.one_hot(states, STATES).float()
        c0 = codes[ops[:, 0]]
        if self.method == "fv_one_shot":
            second = ops[:, 1].clamp_min(0)
            c1 = codes[second]
            c1 = c1 * (ops[:, 1] >= 0).float().unsqueeze(-1)
            joined = torch.cat((onehot @ self.state.weight, c0 + self.phase[0], c1 + self.phase[1]), -1)
            return self.one_shot(joined), None
        logits0 = self._step(onehot, c0, 0)
        active = ops[:, 1] >= 0
        if not bool(active.any()):
            return logits0, None
        if hard_feedback:
            mid = F.one_hot(logits0.argmax(-1), STATES).float()
        else:
            mid = torch.softmax(logits0, dim=-1)
        c1 = codes[ops[:, 1].clamp_min(0)]
        logits1 = self._step(mid, c1, 1)
        out = torch.where(active.unsqueeze(-1), logits1, logits0)
        return out, logits0

    def external_two_call(self, first: torch.Tensor, second: torch.Tensor, states: torch.Tensor,
                          support: torch.Tensor):
        codes = self.function_codes(support)
        distribution = F.one_hot(states.long(), STATES).float()
        first_logits = self._step(distribution, codes[first.long()], 0)
        intermediate = F.one_hot(first_logits.argmax(-1), STATES).float()
        second_logits = self._step(intermediate, codes[second.long()], 1)
        return second_logits, first_logits


def batch_stream(atoms, pairs, seed: int, updates: int, batch_size: int):
    op_a, x_a, y_a = atoms
    ops_p, x_p, mid_p, y_p = pairs
    rng = np.random.default_rng(seed)
    result = []
    half = batch_size // 2
    for _ in range(updates):
        ai = rng.integers(len(x_a), size=half)
        pi = rng.integers(len(x_p), size=batch_size-half)
        ops = np.zeros((batch_size, 2), dtype=np.int64)
        xs = np.concatenate((x_a[ai], x_p[pi]))
        ys = np.concatenate((y_a[ai], y_p[pi]))
        ops[:half, 0] = op_a[ai]; ops[:half, 1] = -1
        ops[half:] = ops_p[pi]
        result.append((ops, xs, ys))
    return result


def train(method, support, atoms, pairs, seed: int, updates: int = 2500):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Executor(method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    sup = torch.as_tensor(support, dtype=torch.long)
    stream = batch_stream(atoms, pairs, seed + 999, updates, 64)
    start = time.perf_counter()
    model.train()
    for ops_np, xs_np, ys_np in stream:
        ops = torch.as_tensor(ops_np); xs = torch.as_tensor(xs_np); ys = torch.as_tensor(ys_np)
        logits, _ = model(ops, xs, sup)
        loss = F.cross_entropy(logits, ys)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model, time.perf_counter() - start


def _metric(model, rows, support, external=False):
    ops = torch.as_tensor(rows[:, :2]); xs = torch.as_tensor(rows[:, 2]); ys = torch.as_tensor(rows[:, 4])
    sup = torch.as_tensor(support)
    model.eval()
    with torch.no_grad():
        logits, first = model.external_two_call(ops[:, 0], ops[:, 1], xs, sup) if external else model(ops, xs, sup, hard_feedback=True)
        pred = logits.argmax(-1)
        nll = float(F.cross_entropy(logits, ys))
        exact = pred.eq(ys)
        if first is None:
            mid_acc = None; path = None
        else:
            mid_acc = float(first.argmax(-1).eq(torch.as_tensor(rows[:, 3])).float().mean())
            path = float((first.argmax(-1).eq(torch.as_tensor(rows[:, 3])) & exact).float().mean())
    return {"exact_pair_accuracy": float(exact.float().mean()), "intermediate_accuracy": mid_acc,
            "intermediate_valid_path_rate": path, "final_token_nll": nll, "examples": len(rows)}


def evaluate(model, support, atoms, held, maps):
    atom_ops, atom_x, atom_y = atoms
    sup = torch.as_tensor(support, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        if model.method == "fv_one_shot":
            atom_inputs = np.stack((atom_ops, np.full_like(atom_ops, -1)), -1)
            logits, _ = model(torch.as_tensor(atom_inputs), torch.as_tensor(atom_x), sup)
        else:
            atom_inputs = np.stack((atom_ops, np.full_like(atom_ops, -1)), -1)
            logits, _ = model(torch.as_tensor(atom_inputs), torch.as_tensor(atom_x), sup, hard_feedback=True)
        atom_acc = float(logits.argmax(-1).eq(torch.as_tensor(atom_y)).float().mean())
    internal = _metric(model, held, support)
    external = _metric(model, held, support, external=True) if model.method in ("fv_tied", "native_tied") else None
    # Reverse-order diagnostic is the two reversed operator pairs in HELDOUT.
    reverse_rows = held[np.isin(held[:, 0], [1, 3])]
    reverse = _metric(model, reverse_rows, support)
    return {"atomic_exact_accuracy": atom_acc, "heldout_pairs": internal,
            "external_two_call": external, "reverse_order": reverse}


def mac_proxy(method: str, batch: int = 1):
    step = (2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES
    if method == "fv_one_shot":
        transition_macs = 3 * WIDTH * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES
    else:
        transition_macs = 2 * step
    # The implementation re-encodes all K support sets per model call. GELU,
    # softmax and integer state lookups are excluded from this MAC proxy.
    support_encoder_macs = K * 8 * (2 * WIDTH * WIDTH) if method.startswith("fv_") else 0
    return int(batch * transition_macs + support_encoder_macs)


def serialize(path: Path, model: Executor, method: str, support: np.ndarray, maps: np.ndarray):
    arrays = {f"weight__{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    arrays["schema"] = np.array([540, K, STATES, WIDTH, HIDDEN], dtype=np.int32)
    arrays["method"] = np.frombuffer(method.encode(), dtype=np.uint8)
    arrays["operator_id_order"] = np.arange(K, dtype=np.uint8)
    if method.startswith("fv_"):
        arrays["support_pairs"] = support.astype(np.uint8)
    np.savez(path, **arrays)
    with zipfile.ZipFile(path) as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_table(path: Path, maps: np.ndarray):
    np.savez(path, operator_tables=maps.astype(np.uint8), schema=np.array([540, K, STATES], dtype=np.int32))
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def runtime(model, support):
    rows = np.array([[0, 1, 3, 0, 0]], dtype=np.int64)
    ops, xs = torch.as_tensor(rows[:, :2]), torch.as_tensor(rows[:, 2])
    sup = torch.as_tensor(support, dtype=torch.long)
    model.eval()
    times = {}
    with torch.no_grad():
        for name, fn in (("internal_unroll", lambda: model(ops, xs, sup, hard_feedback=True)),
                         ("external_two_call", lambda: model.external_two_call(ops[:, 0], ops[:, 1], xs, sup))):
            for _ in range(30): fn()
            samples = []
            for _ in range(150):
                t0 = time.perf_counter(); fn(); samples.append(time.perf_counter() - t0)
            times[name] = float(np.median(samples))
    return times


def run(world: int, out_dir: str):
    torch.set_num_threads(1)
    if world not in DEV_WORLDS + FRESH_WORLDS:
        raise ValueError(f"world {world} is not registered in the frozen protocol")
    if world in FRESH_WORLDS:
        dev_root = ROOT / "runs" / "dev"
        dev_results = [json.loads((dev_root / f"world_{w}" / "metrics.json").read_text()) for w in DEV_WORLDS]
        for dev in dev_results:
            by_name = {row["method"]: row for row in dev["rows"]}
            fv = by_name["fv_tied"]; native = by_name["native_tied"]
            if (fv["metrics"]["heldout_pairs"]["exact_pair_accuracy"] < 0.90
                    or fv["metrics"]["heldout_pairs"]["intermediate_valid_path_rate"] < 0.90
                    or fv["metrics"]["heldout_pairs"]["exact_pair_accuracy"] < native["metrics"]["heldout_pairs"]["exact_pair_accuracy"] - 0.05
                    or fv["payload_bytes"] > 0.90 * native["payload_bytes"]):
                raise RuntimeError("fresh worlds are sealed: the complete amended development gate did not pass")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    maps, support, atoms, pairs, held = make_world(world)
    np.savez(out / "world_inputs.npz", maps=maps.astype(np.uint8), support=support.astype(np.uint8), heldout=held.astype(np.uint8))
    report = {"experiment_id": "MA-540", "world": world, "heldout_pair_list": [list(x) for x in HELDOUT], "rows": []}
    train_macs = {
        "fv_tied": 8 * 8 * (2 * WIDTH * WIDTH) + 32 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES) + 32 * 2 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES),
        "native_tied": 32 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES) + 32 * 2 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES),
        "fv_one_shot": 8 * 8 * (2 * WIDTH * WIDTH) + 64 * (3 * WIDTH * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES),
        "native_untied": 32 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES) + 32 * 2 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES),
    }
    for index, method in enumerate(METHODS):
        model, fit_s = train(method, support, atoms, pairs, world + 1009 * index)
        metrics = evaluate(model, support, atoms, held, maps)
        runtime_s = runtime(model, support) if method in ("fv_tied", "native_tied") else {}
        payload, sha = serialize(out / f"{method}.npz", model, method, support, maps)
        report["rows"].append({"method": method, "payload_bytes": payload, "payload_sha256": sha,
                               "updates": 2500, "train_examples": 2500 * 64,
                               "training_macs_per_update": train_macs[method],
                               "training_macs_total": train_macs[method] * 2500,
                               "active_macs_per_packet": mac_proxy(method), "fit_seconds": fit_s,
                               "runtime_seconds_per_packet": runtime_s, "metrics": metrics})
    size, sha = serialize_table(out / "exact_table_upper.npz", maps)
    report["rows"].append({"method": "exact_operator_table_upper", "payload_bytes": size,
                           "payload_sha256": sha, "updates": 0, "train_examples": 0,
                           "active_macs_per_packet": 2, "fit_seconds": 0,
                           "training_macs_per_update": 0, "training_macs_total": 0,
                           "runtime_seconds_per_packet": {},
                           "metrics": {"atomic_exact_accuracy": 1.0,
                                       "heldout_pairs": {"exact_pair_accuracy": 1.0,
                                                          "intermediate_accuracy": 1.0,
                                                          "intermediate_valid_path_rate": 1.0,
                                                          "final_token_nll": 0.0,
                                                          "examples": len(held)}}})
    for row in report["rows"]:
        row["external_two_call_active_macs_per_packet"] = (
            2 * (K * 8 * (2 * WIDTH * WIDTH)) + 2 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES)
            if row["method"] == "fv_tied" else
            2 * ((2 * WIDTH) * HIDDEN + HIDDEN * HIDDEN + HIDDEN * STATES)
            if row["method"] == "native_tied" else None
        )
    (out / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.world, args.out), indent=2))
