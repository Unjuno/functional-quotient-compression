"""Frozen-feature rank-2 LoRA source-bank models for MA-381."""
from __future__ import annotations

import torch

INPUT_DIM = 16
OUTPUT_DIM = 16
RANK = 2
CANDIDATES = 8
TARGETS = 4


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(2) matrices for (..., 1) angle tensors."""
    theta = angles[..., 0]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack((c, -s, s, c), dim=-1).reshape(*angles.shape[:-1], 2, 2)


class LoRABank(torch.nn.Module):
    METHODS = ("independent", "tied", "scalar", "coeff", "mirror")

    def __init__(self, method: str, seed: int):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(seed)
        if method == "independent":
            self.down = torch.nn.Parameter(torch.randn(CANDIDATES, RANK, INPUT_DIM) / INPUT_DIM**0.5)
            self.up = torch.nn.Parameter(0.3 * torch.randn(CANDIDATES, OUTPUT_DIM, RANK) / RANK**0.5)
        else:
            self.down = torch.nn.Parameter(torch.randn(RANK, INPUT_DIM) / INPUT_DIM**0.5)
            self.up = torch.nn.Parameter(0.3 * torch.randn(OUTPUT_DIM, RANK) / RANK**0.5)
            if method == "scalar":
                self.code = torch.nn.Parameter(torch.ones(CANDIDATES))
            elif method == "coeff":
                self.code = torch.nn.Parameter(torch.eye(RANK).repeat(CANDIDATES, 1, 1))
            elif method == "mirror":
                self.code = torch.nn.Parameter(torch.zeros(CANDIDATES, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "independent":
            h = torch.einsum("bd,nrd->bnr", x, self.down)
            return torch.einsum("bnr,ndr->bnd", h, self.up)
        h = x @ self.down.T
        if self.method == "tied":
            y = h @ self.up.T
            return y[:, None, :].expand(-1, CANDIDATES, -1)
        if self.method == "scalar":
            return (h @ self.up.T)[:, None, :] * self.code[None, :, None]
        if self.method == "coeff":
            z = torch.einsum("br,nkr->bnk", h, self.code)
        else:
            q = rotation_matrices(self.code)
            z = torch.einsum("br,nkr->bnk", h, q)
        return torch.einsum("bnr,dr->bnd", z, self.up)

    def payload_arrays(self, composition: torch.Tensor) -> dict[str, object]:
        arrays: dict[str, object] = {
            "method": self.method,
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "rank": RANK,
            "candidate_count": CANDIDATES,
            "target_count": TARGETS,
            "composition": composition.detach().cpu().numpy().astype("<f4"),
        }
        for name, value in self.state_dict().items():
            arrays[f"bank_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return arrays


def bank_from_payload(payload: dict[str, object]) -> LoRABank:
    method = str(payload["method"])
    bank = LoRABank(method, 0).cpu()
    state = {k.removeprefix("bank_"): torch.as_tensor(v, dtype=torch.float32)
             for k, v in payload.items() if k.startswith("bank_")}
    bank.load_state_dict(state)
    return bank.eval()


def compose(source_outputs: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Signed LoRAHub composition: all target candidate weights are paid state."""
    return torch.einsum("tn,bnd->btd", coefficients, source_outputs)


def mac_proxy(method: str) -> int:
    """Source bank MACs per input; composition summation is reported separately."""
    return {
        "independent": CANDIDATES * 2 * INPUT_DIM * RANK,
        "tied": 2 * INPUT_DIM * RANK,
        "scalar": 2 * INPUT_DIM * RANK + CANDIDATES,
        "coeff": 2 * INPUT_DIM * RANK + CANDIDATES * RANK * RANK + CANDIDATES * OUTPUT_DIM * RANK,
        "mirror": 2 * INPUT_DIM * RANK + CANDIDATES * 4 + CANDIDATES * OUTPUT_DIM * RANK,
    }[method]
