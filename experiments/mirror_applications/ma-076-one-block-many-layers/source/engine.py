"""Small deterministic depth-view regression screen for MA-076."""
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass

import torch
from torch import nn


DEPTH, WIDTH, TRAIN_N, TEST_N = 4, 8, 256, 128
METHODS = ("tied", "static_lora", "generated_gain", "mirror", "untied")


def rotation(theta: torch.Tensor, width: int = WIDTH) -> torch.Tensor:
    """Givens rotation on the first two hidden coordinates."""
    c, s = torch.cos(theta), torch.sin(theta)
    eye = torch.eye(width, dtype=theta.dtype, device=theta.device)
    out = eye.clone()
    out[0, 0], out[0, 1] = c, -s
    out[1, 0], out[1, 1] = s, c
    return out


def make_world(seed: int, aligned: bool):
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(TRAIN_N, WIDTH, generator=g)
    x_test = torch.randn(TEST_N, WIDTH, generator=g)
    base = torch.randn(WIDTH, WIDTH, generator=g) / math.sqrt(WIDTH)
    angles = torch.tensor([-0.45, -0.15, 0.15, 0.45])
    if aligned:
        teachers = torch.stack([rotation(a) @ base @ rotation(-a) for a in angles])
    else:
        teachers = torch.randn(DEPTH, WIDTH, WIDTH, generator=g) / math.sqrt(WIDTH)
    y_train = torch.einsum("dij,nj->dni", teachers, x_train)
    y_test = torch.einsum("dij,nj->dni", teachers, x_test)
    return x_train, y_train, x_test, y_test


class DepthModel(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        if method == "untied":
            self.weight = nn.Parameter(torch.empty(DEPTH, WIDTH, WIDTH))
            nn.init.normal_(self.weight, std=1 / math.sqrt(WIDTH))
        else:
            self.weight = nn.Parameter(torch.empty(WIDTH, WIDTH))
            nn.init.normal_(self.weight, std=1 / math.sqrt(WIDTH))
        if method == "static_lora":
            self.a = nn.Parameter(torch.randn(DEPTH, WIDTH, 1) * 0.05)
            self.b = nn.Parameter(torch.randn(DEPTH, 1, WIDTH) * 0.05)
        elif method == "generated_gain":
            # Shared per-coordinate modulation basis, addressed by depth.
            self.gain = nn.Parameter(torch.ones(DEPTH, WIDTH))
        elif method == "mirror":
            self.angle = nn.Parameter(torch.zeros(DEPTH))

    def matrices(self) -> torch.Tensor:
        if self.method == "untied":
            return self.weight
        if self.method == "static_lora":
            return self.weight.unsqueeze(0) + torch.bmm(self.a, self.b)
        if self.method == "generated_gain":
            return self.gain.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.method == "mirror":
            return torch.stack([rotation(a) @ self.weight @ rotation(-a) for a in self.angle])
        return self.weight.unsqueeze(0).expand(DEPTH, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("dij,nj->dni", self.matrices(), x)


@dataclass
class Result:
    method: str
    seed: int
    aligned: bool
    test_mse: float
    layer_mse_max: float
    payload_bytes: int
    updates: int
    examples: int
    active_macs_proxy: int
    wall_time_s: float
    throughput_examples_s: float


def serialized_payload(model: DepthModel) -> bytes:
    buffer = io.BytesIO()
    # Include architecture/method metadata as inference state.
    torch.save({"method": model.method, "state_dict": model.state_dict()}, buffer)
    return buffer.getvalue()


def serialized_bytes(model: DepthModel) -> int:
    return len(serialized_payload(model))


def train_one(method: str, seed: int, aligned: bool, updates: int = 600, lr: float = 0.01) -> Result:
    torch.manual_seed(seed + 991)
    x, y, xt, yt = make_world(seed, aligned)
    model = DepthModel(method)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(updates):
        pred = model(x)
        loss = (pred - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    with torch.no_grad():
        diff = model(xt) - yt
        per_layer = diff.square().mean(dim=(1, 2))
        mse = float(diff.square().mean())
        layer_max = float(per_layer.max())
    # Dense linear forward/backward proxy plus each method's coordinate construction.
    macs = updates * TRAIN_N * DEPTH * WIDTH * WIDTH * 3
    if method == "mirror":
        # Two dense width-by-width conjugation multiplies per logical layer.
        macs += updates * 3 * 2 * DEPTH * WIDTH**3
    elif method == "static_lora":
        macs += updates * 3 * 2 * DEPTH * WIDTH
    elif method == "generated_gain":
        macs += updates * DEPTH * WIDTH
    macs = int(macs)
    return Result(method, seed, aligned, mse, layer_max, serialized_bytes(model), updates,
                  updates * TRAIN_N, macs, wall, (updates * TRAIN_N) / wall)


def run(seed: int, condition: str, updates: int = 600, lr: float = 0.01) -> list[Result]:
    aligned = condition == "aligned"
    return [train_one(method, seed, aligned, updates, lr) for method in METHODS]
