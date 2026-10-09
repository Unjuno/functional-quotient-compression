"""Quotient/remainder compositional embedding operators for MA-391."""
from __future__ import annotations

import torch

VOCAB = 1024
DIM = 16
Q_ROWS = 32
R_ROWS = 32
CLASSES = 16
METHODS = ("full", "add", "product", "concat", "coeff", "mirror")


def qr_indices(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = token_ids.to(torch.long)
    return torch.div(token_ids, R_ROWS, rounding_mode="floor"), torch.remainder(token_ids, R_ROWS)


class QRBank(torch.nn.Module):
    def __init__(self, method: str, init_seed: int, q_table: torch.Tensor,
                 r_table: torch.Tensor, decoder: torch.Tensor,
                 concat_projection: torch.Tensor, concat_projection_seed: int):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.register_buffer("q_table", q_table.detach().clone().float())
        self.register_buffer("r_table", r_table.detach().clone().float())
        self.register_buffer("decoder", decoder.detach().clone().float())
        self.register_buffer("concat_projection", concat_projection.detach().clone().float())
        self.concat_projection_seed = int(concat_projection_seed)
        torch.manual_seed(init_seed)
        if method == "full":
            self.embedding = torch.nn.Parameter(0.1 * torch.randn(VOCAB, DIM))
        elif method == "coeff":
            self.coeff = torch.nn.Parameter(0.1 * torch.randn(VOCAB, 2))
        elif method == "mirror":
            self.angle = torch.nn.Parameter(2 * torch.pi * torch.rand(VOCAB))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.method == "full":
            return self.embedding[token_ids]
        qi, ri = qr_indices(token_ids)
        q = self.q_table[qi]
        r = self.r_table[ri]
        if self.method == "add":
            return q + r
        if self.method == "product":
            return q * r
        if self.method == "concat":
            return torch.cat((q, r), dim=-1) @ self.concat_projection
        if self.method == "coeff":
            c = self.coeff[token_ids]
            return c[..., 0:1] * q + c[..., 1:2] * r
        if self.method == "mirror":
            a = self.angle[token_ids]
            return torch.cos(a)[..., None] * q + torch.sin(a)[..., None] * r
        raise AssertionError(self.method)

    def payload_arrays(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "vocabulary_size": VOCAB,
            "embedding_dim": DIM,
            "quotient_rows": Q_ROWS,
            "remainder_rows": R_ROWS,
            "decoder": self.decoder.detach().cpu().numpy().astype("<f4"),
        }
        if self.method != "full":
            payload["q_table"] = self.q_table.detach().cpu().numpy().astype("<f4")
            payload["r_table"] = self.r_table.detach().cpu().numpy().astype("<f4")
        if self.method == "concat":
            payload["concat_projection"] = self.concat_projection.detach().cpu().numpy().astype("<f4")
            payload["concat_projection_seed"] = self.concat_projection_seed
        for name, value in self.state_dict().items():
            if name in {"q_table", "r_table", "decoder", "concat_projection"}:
                continue
            payload[f"state_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return payload


def from_payload(payload: dict[str, object]) -> QRBank:
    method = str(payload["method"])
    q = torch.as_tensor(payload.get("q_table", torch.zeros(Q_ROWS, DIM)), dtype=torch.float32)
    r = torch.as_tensor(payload.get("r_table", torch.zeros(R_ROWS, DIM)), dtype=torch.float32)
    decoder = torch.as_tensor(payload["decoder"], dtype=torch.float32)
    projection = torch.as_tensor(payload.get("concat_projection", torch.zeros(2 * DIM, DIM)), dtype=torch.float32)
    projection_seed = int(payload.get("concat_projection_seed", 0))
    bank = QRBank(method, 0, q, r, decoder, projection, projection_seed)
    state = {key.removeprefix("state_"): torch.as_tensor(value, dtype=torch.float32)
             for key, value in payload.items() if key.startswith("state_")}
    if method == "full":
        bank.load_state_dict(state, strict=False)
    elif method == "coeff":
        bank.load_state_dict(state, strict=False)
    elif method == "mirror":
        bank.load_state_dict(state, strict=False)
    return bank.eval()


def compute_proxy(method: str) -> dict[str, int]:
    return {
        "component_row_lookups": 0 if method == "full" else 2,
        "composition_macs_per_token": {
            "full": 0, "add": DIM, "product": DIM, "concat": 2 * DIM * DIM,
            "coeff": 2 * DIM, "mirror": 2 * DIM
        }[method],
        "mirror_trig_ops_per_token": 2 if method == "mirror" else 0,
        "fixed_decoder_macs_per_token": DIM * CLASSES,
    }
