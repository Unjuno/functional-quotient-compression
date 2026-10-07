"""MS008 residual sensor-view operator.

Sensor identity is used only to address a low-dimensional residual View.  The
shared world core itself receives canonical state and no sensor ID.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ResidualView(nn.Module):
    def __init__(self, width: int, geometry: str, rank: int, alpha: float,
                 seed: int, code_max: float = 1.0):
        super().__init__()
        self.width = width
        self.geometry = geometry
        self.rank = rank
        self.alpha = float(alpha)
        self.code_max = float(code_max)
        self.raw_code = nn.Parameter(torch.zeros(4, rank))

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(width, generator=g)
        self.register_buffer("src", perm[:rank], persistent=False)
        self.register_buffer("dst", perm[rank:2 * rank], persistent=False)

    def codes(self, address: torch.Tensor) -> torch.Tensor:
        c = self.code_max * torch.tanh(self.raw_code[address])
        # sensor0 is the canonical reference path
        return c * (address != 0).to(c.dtype)[:, None]

    def _shear(self, x, c, inverse=False):
        y = x.clone()
        sign = -1.0 if inverse else 1.0
        y[:, self.dst] = y[:, self.dst] + sign * c * x[:, self.src]
        return y

    def _stretch(self, x, c, inverse=False):
        y = x.clone()
        sign = -1.0 if inverse else 1.0
        scale = torch.exp(sign * c)
        y[:, self.src] = y[:, self.src] * scale
        y[:, self.dst] = y[:, self.dst] / scale
        return y

    def transform(self, x, c):
        if self.geometry == "shear":
            return self._shear(x, c)
        if self.geometry == "stretch":
            return self._stretch(x, c)
        if self.geometry == "combined":
            return self._shear(self._stretch(x, c), c)
        raise ValueError(self.geometry)

    def inverse(self, x, c):
        if self.geometry == "shear":
            return self._shear(x, c, True)
        if self.geometry == "stretch":
            return self._stretch(x, c, True)
        if self.geometry == "combined":
            return self._stretch(self._shear(x, c, True), c, True)
        raise ValueError(self.geometry)

    def forward(self, x, address):
        base = F.gelu(x, approximate="tanh")
        if self.alpha == 0:
            return base
        c = self.codes(address)
        mirrored = self.inverse(
            F.gelu(self.transform(x, c), approximate="tanh"), c
        )
        return base + self.alpha * (mirrored - base)
