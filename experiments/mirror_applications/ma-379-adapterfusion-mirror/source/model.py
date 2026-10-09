"""Small frozen-feature AdapterFusion proxy and source adapter bank variants."""
from __future__ import annotations

import itertools

import torch


INPUT_DIM = 16
OUTPUT_DIM = 16
RANK = 4
SOURCES = 8
TARGETS = 4
PLANES = tuple(itertools.combinations(range(RANK), 2))


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Compose six Givens rotations; angles has shape (..., 6)."""
    lead = angles.shape[:-1]
    eye = torch.eye(RANK, dtype=angles.dtype, device=angles.device)
    result = eye.expand(*lead, RANK, RANK).clone()
    for idx, (i, j) in enumerate(PLANES):
        theta = angles[..., idx]
        c, s = torch.cos(theta), torch.sin(theta)
        g = eye.expand(*lead, RANK, RANK).clone()
        g[..., i, i] = c
        g[..., j, j] = c
        g[..., i, j] = -s
        g[..., j, i] = s
        result = result @ g
    return result


class SourceBank(torch.nn.Module):
    """Independent, tied, scalar, generic-coordinate, or Givens adapter bank."""

    METHODS = ("independent", "tied", "scalar", "coeff", "mirror")

    def __init__(self, method: str, seed: int):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(seed)
        if method == "independent":
            self.down = torch.nn.Parameter(torch.randn(SOURCES, RANK, INPUT_DIM) / INPUT_DIM**0.5)
            self.up = torch.nn.Parameter(0.3 * torch.randn(SOURCES, OUTPUT_DIM, RANK) / RANK**0.5)
        else:
            self.down = torch.nn.Parameter(torch.randn(RANK, INPUT_DIM) / INPUT_DIM**0.5)
            self.up = torch.nn.Parameter(0.3 * torch.randn(OUTPUT_DIM, RANK) / RANK**0.5)
            if method == "scalar":
                self.code = torch.nn.Parameter(torch.ones(SOURCES))
            elif method == "coeff":
                self.code = torch.nn.Parameter(torch.eye(RANK).repeat(SOURCES, 1, 1))
            elif method == "mirror":
                self.code = torch.nn.Parameter(torch.zeros(SOURCES, len(PLANES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "independent":
            h = torch.einsum("bd,nrd->bnr", x, self.down)
            return torch.einsum("bnr,ndr->bnd", h, self.up)

        h = x @ self.down.T
        if self.method == "tied":
            y = h @ self.up.T
            return y[:, None, :].expand(-1, SOURCES, -1)
        if self.method == "scalar":
            y = h @ self.up.T
            return y[:, None, :] * self.code[None, :, None]
        if self.method == "coeff":
            z = torch.einsum("br,nkr->bnk", h, self.code)
        else:
            q = rotation_matrices(self.code)
            z = torch.einsum("br,nkr->bnk", h, q)
        return torch.einsum("bnr,dr->bnd", z, self.up)

    def payload_arrays(self, router: torch.Tensor) -> dict[str, object]:
        arrays: dict[str, object] = {
            "method": self.method,
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "rank": RANK,
            "source_count": SOURCES,
            "target_count": TARGETS,
            "router": router.detach().cpu().numpy().astype("<f4"),
        }
        for name, value in self.state_dict().items():
            arrays[f"bank_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return arrays


def bank_from_payload(payload: dict[str, object], device: str = "cpu") -> SourceBank:
    method = str(payload["method"])
    bank = SourceBank(method, 0).to(device)
    state = {k.removeprefix("bank_"): torch.as_tensor(v, dtype=torch.float32, device=device)
             for k, v in payload.items() if k.startswith("bank_")}
    bank.load_state_dict(state)
    return bank


def target_router(x: torch.Tensor, router: torch.Tensor, source_outputs: torch.Tensor) -> torch.Tensor:
    """Input-conditioned AdapterFusion-style softmax over source adapter outputs."""
    weights = torch.softmax(x @ router, dim=-1)
    return torch.einsum("bn,bnd->bd", weights, source_outputs)


def mac_proxy(method: str) -> int:
    """Multiply-add proxy for source bank plus one target fusion; arithmetic ops excluded."""
    source = {
        "independent": SOURCES * 2 * INPUT_DIM * RANK,
        "tied": 2 * INPUT_DIM * RANK,
        "scalar": 2 * INPUT_DIM * RANK + SOURCES,
        "coeff": 2 * INPUT_DIM * RANK + SOURCES * RANK * RANK + SOURCES * OUTPUT_DIM * RANK,
        "mirror": 2 * INPUT_DIM * RANK + SOURCES * 4 * len(PLANES) + SOURCES * OUTPUT_DIM * RANK,
    }[method]
    return source
