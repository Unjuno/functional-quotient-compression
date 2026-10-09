"""MA-431 controlled screen: step-addressed views of one recurrent block."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DIM = 16
MAX_LEN = 6
BATCH = 128
METHODS = ("static", "rank1", "universal_lowrank", "mirror", "independent")


def op_matrix(symbol: int, shear: float, *, device: str = "cpu") -> torch.Tensor:
    m = torch.eye(DIM, dtype=torch.float32, device=device)
    if symbol == 0:
        m[0, 1] = shear
    else:
        m[1, 0] = -shear
    return m


def mirror_matrix(w: torch.Tensor, symbol: int, coordinate_scale: float = 1.0) -> torch.Tensor:
    """R(m) W R(-m), with a deterministic 0 or pi/2 Givens address."""
    if symbol == 0:
        return w
    # Apply R(m) on the leading two rows and R(-m) on the columns in O(D),
    # instead of materializing two dense matrix products.
    angle = coordinate_scale * math.pi / 2
    c, s = math.cos(angle), math.sin(angle)
    left = w.clone()
    row0 = c * w[0] - s * w[1]
    row1 = s * w[0] + c * w[1]
    left[0] = row0
    left[1] = row1
    out = left.clone()
    col0 = c * left[:, 0] - s * left[:, 1]
    col1 = s * left[:, 0] + c * left[:, 1]
    out[:, 0] = col0
    out[:, 1] = col1
    return out


def teacher_output(x: torch.Tensor, program: tuple[int, ...], shear: float) -> torch.Tensor:
    y = x
    for symbol in program:
        y = op_matrix(symbol, shear, device=x.device) @ y
    return y


def heldout_programs() -> list[tuple[int, ...]]:
    # Hold out all length-six strings whose first and last symbols differ and
    # whose interior contains exactly two 1s. The rule is fixed pre-run.
    out = []
    for p in itertools.product((0, 1), repeat=MAX_LEN):
        if p[0] != p[-1] and sum(p[1:-1]) == 2:
            out.append(p)
    return out


@dataclass
class SharedModels:
    method: str
    base: nn.Parameter
    rank1_u: nn.Parameter | None = None
    rank1_v: nn.Parameter | None = None
    low_u: nn.Parameter | None = None
    low_v: nn.Parameter | None = None
    symbol_codes: nn.Parameter | None = None
    depth_codes: nn.Parameter | None = None
    independent: nn.Parameter | None = None

    def parameters(self):
        params = [] if self.method == "independent" else [self.base]
        return params + [
            p for key, p in self.__dict__.items()
            if key != "base" and isinstance(p, nn.Parameter) and p.requires_grad
        ]

    def matrix(self, symbol: int, depth: int) -> torch.Tensor:
        if self.method == "independent":
            assert self.independent is not None
            return self.independent[symbol]
        if self.method == "mirror":
            return mirror_matrix(self.base, symbol, getattr(self, "_coordinate_scale", 1.0))
        if self.method == "static":
            return self.base
        if self.method == "rank1":
            assert self.rank1_u is not None and self.rank1_v is not None
            coeff = -1.0 if symbol == 0 else 1.0
            return self.base + coeff * torch.outer(self.rank1_u, self.rank1_v)
        if self.method == "universal_lowrank":
            assert all(p is not None for p in (self.low_u, self.low_v, self.symbol_codes, self.depth_codes))
            code = torch.tanh(self.symbol_codes[symbol] + self.depth_codes[depth])
            return self.base + self.low_u @ torch.diag(code) @ self.low_v.T
        raise ValueError(self.method)

    def forward(self, x: torch.Tensor, program: tuple[int, ...]) -> torch.Tensor:
        y = x
        for depth, symbol in enumerate(program):
            y = self.matrix(symbol, depth) @ y
        return y

    def forward_batch(self, x: torch.Tensor, programs: list[tuple[int, ...]]) -> torch.Tensor:
        y = x
        max_depth = max(map(len, programs))
        for depth in range(max_depth):
            active = torch.tensor([len(p) > depth for p in programs], dtype=torch.bool)
            symbols = torch.tensor([p[depth] if len(p) > depth else 0 for p in programs])
            next_y = y.clone()
            for symbol in (0, 1):
                mask = active & (symbols == symbol)
                if mask.any():
                    next_y[mask] = y[mask] @ self.matrix(symbol, depth).T
            y = next_y
        return y


def make_model(method: str, seed: int) -> SharedModels:
    g = torch.Generator(device="cpu").manual_seed(seed)
    eye = torch.eye(DIM, dtype=torch.float32)
    base = nn.Parameter(eye.clone())
    if method == "static" or method == "mirror":
        return SharedModels(method=method, base=base)
    if method == "rank1":
        u = nn.Parameter(torch.randn(DIM, generator=g) * 0.02)
        v = nn.Parameter(torch.randn(DIM, generator=g) * 0.02)
        return SharedModels(method=method, base=base, rank1_u=u, rank1_v=v)
    if method == "universal_lowrank":
        u = nn.Parameter(torch.randn(DIM, 2, generator=g) * 0.02)
        v = nn.Parameter(torch.randn(DIM, 2, generator=g) * 0.02)
        sc = nn.Parameter(torch.zeros(2, 2))
        dc = nn.Parameter(torch.zeros(MAX_LEN, 2))
        return SharedModels(method=method, base=base, low_u=u, low_v=v, symbol_codes=sc, depth_codes=dc)
    if method == "independent":
        base = nn.Parameter(torch.empty(0), requires_grad=False)
        mats = nn.Parameter(eye.repeat(2, 1, 1))
        return SharedModels(method=method, base=base, independent=mats)
    raise ValueError(method)


def model_state(model: SharedModels) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    if model.method != "independent":
        state["base"] = model.base.detach().cpu()
    for key in ("rank1_u", "rank1_v", "low_u", "low_v", "symbol_codes", "depth_codes", "independent"):
        value = getattr(model, key)
        if value is not None:
            state[key] = value.detach().cpu()
    return state


def inference_payload(model: SharedModels) -> bytes:
    metadata = {
        "method": model.method,
        "dimension": DIM,
        "max_len": MAX_LEN,
        "coordinate_scale": getattr(model, "_coordinate_scale", 1.0),
        "operation_codebook": {"0": 0.0, "1": getattr(model, "_coordinate_scale", 1.0) * math.pi / 2},
        "codebook_rule": "symbol 0 -> identity chart; symbol 1 -> leading-plane Givens pi/2 conjugation",
    }
    buf = io.BytesIO()
    torch.save({"state_dict": model_state(model), "metadata": metadata}, buf)
    return buf.getvalue()


def payload_bytes(model: SharedModels) -> tuple[int, str]:
    payload = inference_payload(model)
    return len(payload), hashlib.sha256(payload).hexdigest()


def active_macs_per_example(method: str, length: int) -> int:
    # Count dense matrix-vector MACs and explicit rank/view work. This is a
    # declared eager-operation proxy, not hardware FLOPs.
    dense = DIM * DIM * length
    if method == "independent":
        return dense
    if method == "universal_lowrank":
        # Shared rank-2 update construction plus its recurrent matvec.
        return dense + (2 * DIM * DIM * 2 + 2 * DIM * 2) * length
    if method == "rank1":
        # Materialize one shared outer-product residual and apply it.
        return dense + (DIM * DIM + 2 * DIM) * length
    if method == "mirror":
        # O(D) row/column Givens conjugation plus the shared matvec.
        return dense + 8 * DIM * length
    return dense


def draw_batch(
    rng: random.Random,
    programs: list[tuple[int, ...]],
    batch_size: int,
    shear: float,
) -> tuple[torch.Tensor, list[tuple[int, ...]], torch.Tensor]:
    chosen = [rng.choice(programs) for _ in range(batch_size)]
    x = torch.randn(batch_size, DIM, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))
    y = x.clone()
    max_depth = max(map(len, chosen))
    teacher_ops = (op_matrix(0, shear), op_matrix(1, shear))
    for depth in range(max_depth):
        active = torch.tensor([len(p) > depth for p in chosen], dtype=torch.bool)
        symbols = torch.tensor([p[depth] if len(p) > depth else 0 for p in chosen])
        next_y = y.clone()
        for symbol in (0, 1):
            mask = active & (symbols == symbol)
            if mask.any():
                next_y[mask] = y[mask] @ teacher_ops[symbol].T
        y = next_y
    return x, chosen, y


def all_programs() -> list[tuple[int, ...]]:
    test = set(heldout_programs())
    return [p for length in range(1, MAX_LEN + 1) for p in itertools.product((0, 1), repeat=length) if p not in test]


def train_one(
    method: str,
    world: int,
    seed: int,
    learning_rate: float = 0.003,
    coordinate_scale: float = 1.0,
    updates: int = 1500,
) -> dict[str, Any]:
    # Each world changes the function family, while seed changes minibatches and
    # evaluation inputs. Fresh worlds are never read by this function unless
    # explicitly selected by the caller after development passes.
    torch.set_num_threads(1)
    random.seed(world * 1_000 + seed)
    shear = 0.18 + 0.06 * ((world % 7) / 6.0)
    model = make_model(method, world * 100 + seed)
    if method == "mirror":
        # Coordinate scale is fixed by development selection; codes remain
        # deterministic and paid in metadata.
        # Interpolating angle scales tests whether the registered Givens chart
        # is robust to a compact scalar code.
        model._coordinate_scale = coordinate_scale
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    programs = all_programs()
    rng = random.Random(world * 10_000 + seed)
    start = time.perf_counter()
    examples = 0
    for _ in range(updates):
        x, ps, y = draw_batch(rng, programs, BATCH, shear)
        preds = model.forward_batch(x, ps)
        loss = ((preds - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        examples += BATCH
    train_seconds = time.perf_counter() - start

    # Evaluation is on a fixed, pre-registered held-out set of depth-6 strings
    # and new input vectors. This is development when world<43110 and fresh
    # otherwise; the caller enforces the fresh gate.
    eval_rng = random.Random(8_431_000 + world * 100 + seed)
    test_ps = heldout_programs()
    x = torch.randn(len(test_ps) * 32, DIM, generator=torch.Generator().manual_seed(eval_rng.randrange(1 << 30)))
    eval_programs = [test_ps[i // 32] for i in range(len(x))]
    targets = torch.stack([teacher_output(x[i], eval_programs[i], shear) for i in range(len(x))])
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.forward_batch(x, eval_programs)
    inference_seconds = time.perf_counter() - t0
    relative_mse = float(((preds - targets).square().sum() / targets.square().sum()).item())
    pbytes, sha = payload_bytes(model)
    return {
        "world": world,
        "seed": seed,
        "method": method,
        "learning_rate": learning_rate,
        "coordinate_scale": coordinate_scale if method == "mirror" else 1.0,
        "program_length": MAX_LEN,
        "relative_mse": relative_mse,
        "payload_bytes": pbytes,
        "active_macs_proxy": active_macs_per_example(method, MAX_LEN),
        "train_examples": examples,
        "updates": updates,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "examples_per_second": len(x) / inference_seconds if inference_seconds else float("inf"),
        "sha256": sha,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        summary[method] = {
            key: sum(float(r[key]) for r in subset) / len(subset)
            for key in ("relative_mse", "payload_bytes", "active_macs_proxy", "train_seconds", "inference_seconds", "examples_per_second")
        }
    return summary


def main() -> None:
    import argparse
    import csv
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, nargs="+", default=[43100, 43101])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.003])
    parser.add_argument("--coordinate-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for world in args.worlds:
        for seed in args.seeds:
            for method in args.methods:
                scales = args.coordinate_scales if method == "mirror" else [1.0]
                for lr in args.lrs:
                    for scale in scales:
                        row = train_one(method, world, seed, lr, scale, args.updates)
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
