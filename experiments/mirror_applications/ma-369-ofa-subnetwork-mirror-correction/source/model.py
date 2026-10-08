from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCHS = ((16, 1), (16, 2), (32, 1), (32, 2), (64, 1), (64, 2))
METHODS = ("ofa_shared", "ofa_mirror", "ofa_film", "ofa_lora", "independent")


def rotate_first_pairs(h: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply one Givens angle to each of the first four hidden pairs."""
    pairs = h[:, :8].reshape(h.shape[0], 4, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    c, s = torch.cos(angles).unsqueeze(0), torch.sin(angles).unsqueeze(0)
    rotated = torch.stack((c * a - s * b, s * a + c * b), dim=-1).reshape(h.shape[0], 8)
    return torch.cat((rotated, h[:, 8:]), dim=1)


def apply_code(h: torch.Tensor, method: str, code: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if code is None or method == "ofa_shared":
        return h
    if method == "ofa_mirror":
        return rotate_first_pairs(h, code["angles"])
    if method == "ofa_film":
        scaled = h[:, :8] * code["scales"].repeat_interleave(2)
        return torch.cat((scaled, h[:, 8:]), dim=1)
    return h


class OFASupernet(nn.Module):
    """Nested-width MLP with one or two hidden layers."""
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(64, 64))
        self.b1 = nn.Parameter(torch.zeros(64))
        self.w2 = nn.Parameter(torch.empty(64, 64))
        self.b2 = nn.Parameter(torch.zeros(64))
        self.head = nn.Parameter(torch.empty(10, 64))
        self.head_bias = nn.Parameter(torch.zeros(10))
        nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.head, a=5 ** 0.5)

    def hidden(self, x: torch.Tensor, width: int, depth: int) -> torch.Tensor:
        h = F.relu(F.linear(x, self.w1[:width, :], self.b1[:width]))
        if depth == 2:
            h = F.relu(F.linear(h, self.w2[:width, :width], self.b2[:width]))
        return h

    def forward(self, x: torch.Tensor, arch: tuple[int, int], method: str = "ofa_shared",
                code: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        width, depth = arch
        h = self.hidden(x, width, depth)
        h = apply_code(h, method, code)
        return F.linear(h, self.head[:, :width], self.head_bias)


class IndependentSubnet(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.fc1 = nn.Linear(64, width)
        self.fc2 = nn.Linear(width, width) if depth == 2 else None
        self.head = nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        if self.fc2 is not None:
            h = F.relu(self.fc2(h))
        return self.head(h)


def macs_per_example(arch: tuple[int, int]) -> int:
    width, depth = arch
    return 64 * width + (width * width if depth == 2 else 0) + width * 10


def correction_macs_per_example(method: str, arch: tuple[int, int]) -> int:
    width, _ = arch
    if method == "ofa_mirror":
        return 24
    if method == "ofa_film":
        return 8
    if method == "ofa_lora":
        return width + 10
    return 0
