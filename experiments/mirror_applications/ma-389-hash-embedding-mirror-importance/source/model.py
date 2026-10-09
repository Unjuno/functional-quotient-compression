"""Hash Embedding baseline and per-token importance parameterizations."""
from __future__ import annotations

import torch

VOCAB = 256
DIM = 16
BUCKETS = 32
TABLES = 2
CLASSES = 16
MASK64 = (1 << 64) - 1


def hash_bucket(token: int, seed: int, buckets: int = BUCKETS) -> int:
    """Specified SplitMix64-based hash; stable across Python processes/platforms."""
    x = (int(token) ^ int(seed)) & MASK64
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x = x ^ (x >> 31)
    return int(x % buckets)


def hash_indices(seed0: int, seed1: int) -> torch.Tensor:
    return torch.tensor([[hash_bucket(i, seed0), hash_bucket(i, seed1)] for i in range(VOCAB)], dtype=torch.long)


class HashEmbeddingBank(torch.nn.Module):
    METHODS = ("full", "hash", "tied", "scalar", "mirror")

    def __init__(self, method: str, init_seed: int, hash_seed0: int, hash_seed1: int,
                 decoder: torch.Tensor):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method
        self.hash_seed0, self.hash_seed1 = int(hash_seed0), int(hash_seed1)
        self.indices = hash_indices(hash_seed0, hash_seed1)
        self.register_buffer("decoder", decoder.detach().clone().float())
        torch.manual_seed(init_seed)
        if method == "full":
            self.embedding = torch.nn.Parameter(0.1 * torch.randn(VOCAB, DIM))
        else:
            self.component = torch.nn.Parameter(0.1 * torch.randn(TABLES, BUCKETS, DIM))
            if method == "hash":
                self.importance = torch.nn.Parameter(0.1 * torch.randn(VOCAB, TABLES))
            elif method == "tied":
                self.importance = torch.nn.Parameter(torch.ones(TABLES))
            elif method == "scalar":
                self.base_importance = torch.nn.Parameter(torch.ones(TABLES))
                self.scalar = torch.nn.Parameter(torch.ones(VOCAB))
            else:
                self.angle = torch.nn.Parameter(2 * torch.pi * torch.rand(VOCAB))

    def importance_values(self) -> torch.Tensor:
        if self.method == "hash":
            return self.importance
        if self.method == "tied":
            return self.importance[None, :].expand(VOCAB, -1)
        if self.method == "scalar":
            return self.scalar[:, None] * self.base_importance[None, :]
        if self.method == "mirror":
            return torch.stack((torch.cos(self.angle), torch.sin(self.angle)), dim=-1)
        raise ValueError("full embedding has no hash importance weights")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.method == "full":
            return self.embedding[token_ids]
        idx = self.indices[token_ids]
        v0 = self.component[0, idx[..., 0]]
        v1 = self.component[1, idx[..., 1]]
        alpha = self.importance_values()[token_ids]
        return alpha[..., 0:1] * v0 + alpha[..., 1:2] * v1

    def payload_arrays(self) -> dict[str, object]:
        arrays: dict[str, object] = {
            "method": self.method,
            "vocabulary_size": VOCAB,
            "embedding_dim": DIM,
            "bucket_count": BUCKETS,
            "table_count": TABLES,
            "decoder": self.decoder.detach().cpu().numpy().astype("<f4"),
        }
        if self.method != "full":
            arrays["hash_seeds"] = torch.tensor([self.hash_seed0, self.hash_seed1], dtype=torch.int64).numpy()
        for name, value in self.state_dict().items():
            if name == "decoder":
                continue
            arrays[f"state_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return arrays


def bank_from_payload(payload: dict[str, object]) -> HashEmbeddingBank:
    method = str(payload["method"])
    seeds = np_to_ints(payload.get("hash_seeds", [0, 0]))
    decoder = torch.as_tensor(payload["decoder"], dtype=torch.float32)
    bank = HashEmbeddingBank(method, 0, seeds[0], seeds[1], decoder)
    state = {k.removeprefix("state_"): torch.as_tensor(v, dtype=torch.float32)
             for k, v in payload.items() if k.startswith("state_")}
    state["decoder"] = decoder
    bank.load_state_dict(state)
    return bank.eval()


def np_to_ints(values: object) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def compute_proxy(method: str) -> dict[str, int]:
    return {
        "component_vector_lookups_per_token": 0 if method == "full" else 2,
        "hash_integer_ops_per_token": 0 if method == "full" else 16,
        "importance_reconstruction_ops_per_token": {
            "full": 0, "hash": 0, "tied": 0, "scalar": 2, "mirror": 2
        }[method],
        "fixed_decoder_macs_per_token": DIM * CLASSES,
    }
