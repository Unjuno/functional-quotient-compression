"""Small causal LM adapter for MA-784; vendor nanoGPT files stay untouched."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("ma784_nanogpt_model", REPO / "third_party" / "nanoGPT" / "model.py")
_nano = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_nano)

CONDITIONS = ("dense", "untied_moe", "unipool4", "unipool2", "hard_alias2x", "mirror2x", "film2x")
SHAPES = {
    "dense": (1, 1, "dense"),
    "untied_moe": (4, 4, "untied"),
    "unipool4": (4, 4, "shared"),
    "unipool2": (2, 2, "shared"),
    "hard_alias2x": (2, 4, "shared"),
    "mirror2x": (2, 4, "mirror"),
    "film2x": (2, 4, "film"),
}


def estimate_normrouter_c(experts: int, top_k: int = 1, seed: int = 78400, samples: int = 100_000) -> float:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(samples, experts, generator=gen)
    y = F.relu(z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8))
    top = torch.topk(y, top_k, dim=-1).values
    norm = torch.linalg.vector_norm(top, dim=-1)
    return float(experts / top_k * torch.reciprocal(norm[norm > 0]).mean())


class Expert(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(width, hidden)
        self.c_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


def apply_givens(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs = torch.arange(16, device=x.device).reshape(8, 2)
    ids = pairs.reshape(-1)
    selected = x.index_select(-1, ids).reshape(*x.shape[:-1], 8, 2)
    left, right = selected[..., 0], selected[..., 1]
    if angles.ndim == 1:
        shape = (1,) * (x.ndim - 1) + (8,)
        c, s = torch.cos(angles).view(shape), torch.sin(angles).view(shape)
    else:
        c, s = torch.cos(angles), torch.sin(angles)
    rotated = torch.stack((c * left - s * right, s * left + c * right), dim=-1).reshape(*x.shape[:-1], 16)
    out = x.clone()
    out[..., ids] = rotated
    return out


class RoutedLayer(nn.Module):
    def __init__(self, config, layer_id: int, condition: str):
        super().__init__()
        self.layer_id = layer_id
        self.condition = condition
        physical, logical, mode = SHAPES[condition]
        self.physical, self.logical, self.mode = physical, logical, mode
        cfg = SimpleNamespace(n_embd=config.width, n_head=config.heads, block_size=config.block_size,
                              dropout=config.dropout, bias=True)
        self.ln_1 = _nano.LayerNorm(config.width, bias=True)
        self.attn = _nano.CausalSelfAttention(cfg)
        self.ln_2 = _nano.LayerNorm(config.width, bias=True)
        self.router = nn.Linear(config.width, logical) if mode != "dense" else None
        self.router_scale = nn.Parameter(torch.ones(())) if mode != "dense" else None
        self.local_experts = nn.ModuleList([Expert(config.width, config.expert_hidden, config.dropout)
                                            for _ in range(physical)]) if mode == "untied" else None
        self.dense_mlp = Expert(config.width, config.expert_hidden, config.dropout) if mode == "dense" else None

    def forward(self, x, shared_experts, view_state, config):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        if self.mode == "dense":
            return x + self.dense_mlp(h), h.new_zeros((1,)), h.new_zeros((1,)), h.new_zeros(())
        logits = self.router(h)
        normalized = logits / torch.linalg.vector_norm(logits, dim=-1, keepdim=True).clamp_min(1e-8)
        scale_c = config.normrouter_c_by_logical[self.logical]
        scores = F.relu(normalized) * scale_c * self.router_scale
        top_score, logical_id = scores.max(dim=-1)
        physical_id = logical_id if self.mode == "untied" else logical_id.remainder(self.physical)
        flat_h, flat_idx, flat_score = h.reshape(-1, h.shape[-1]), physical_id.reshape(-1), top_score.reshape(-1, 1)
        flat_route = logical_id.reshape(-1)
        flat_out = torch.zeros_like(flat_h)
        bank = self.local_experts if self.mode == "untied" else shared_experts
        for expert_id, expert in enumerate(bank):
            token_ids = torch.nonzero(flat_idx == expert_id, as_tuple=False).squeeze(-1)
            if not token_ids.numel():
                continue
            values = flat_h.index_select(0, token_ids)
            if self.mode == "mirror":
                route_ids = flat_route.index_select(0, token_ids)
                angles = view_state["givens"][self.layer_id].index_select(0, route_ids)
                values = apply_givens(values, angles)
            elif self.mode == "film":
                route_ids = flat_route.index_select(0, token_ids)
                scales = view_state["film_scale"][self.layer_id].index_select(0, route_ids)
                channel_groups = torch.arange(config.width, device=h.device) // (config.width // config.gate_groups)
                values = values * scales[:, channel_groups]
            out = expert(values) * flat_score.index_select(0, token_ids)
            flat_out = flat_out.index_copy(0, token_ids, out)
        logical_probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        assignments = F.one_hot(logical_id, num_classes=self.logical).to(scores.dtype)
        physical_importance = scores.new_zeros((self.physical,))
        physical_load = scores.new_zeros((self.physical,))
        for route_id in range(self.logical):
            physical_id = route_id if self.mode == "untied" else route_id % self.physical
            physical_importance[physical_id] += scores[..., route_id].mean()
            physical_load[physical_id] += assignments[..., route_id].mean()
        entropy = -(logical_probs * logical_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return x + flat_out.reshape_as(x), physical_importance, physical_load.detach(), entropy


class SmallGPT(nn.Module):
    def __init__(self, config, condition: str, vocab_size: int):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.config, self.condition = config, condition
        physical, _, mode = SHAPES[condition]
        self.wte = nn.Embedding(vocab_size, config.width)
        self.wpe = nn.Embedding(config.block_size, config.width)
        self.shared_experts = nn.ModuleList([Expert(config.width, config.expert_hidden, config.dropout)
                                             for _ in range(physical)]) if mode == "shared" or mode in ("mirror", "film") else None
        self.layers = nn.ModuleList([RoutedLayer(config, i, condition) for i in range(config.layers)])
        self.ln_f = _nano.LayerNorm(config.width, bias=True)
        self.lm_head = nn.Linear(config.width, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        if mode == "mirror":
            self.givens = nn.Parameter(torch.zeros(config.layers, 4, 8))
        else:
            self.register_parameter("givens", None)
        if mode == "film":
            self.film_scale = nn.Parameter(torch.ones(config.layers, 4, config.gate_groups))
        else:
            self.register_parameter("film_scale", None)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.layers) ** 0.5)
        if self.givens is not None:
            nn.init.zeros_(self.givens)
        if self.film_scale is not None:
            nn.init.ones_(self.film_scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        view = {"givens": self.givens, "film_scale": self.film_scale}
        importance, loads, entropies = [], [], []
        for layer in self.layers:
            x, imp, load, entropy = layer(x, self.shared_experts, view, self.config)
            if imp.numel() > 1:
                importance.append(imp); loads.append(load); entropies.append(entropy)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) if targets is not None else None
        if importance:
            imp, load = torch.stack(importance).mean(0), torch.stack(loads).mean(0)
            aux = len(imp) * (imp * load).sum()
            mean_load, entropy = load, torch.stack(entropies).mean()
        else:
            aux = logits.new_zeros(()); mean_load = logits.new_zeros((1,)); entropy = logits.new_zeros(())
        return logits, loss, aux, mean_load, entropy


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
