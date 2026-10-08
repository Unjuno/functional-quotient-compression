"""Thin nanoGPT attention adapter with routed/shared expert variants."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("ma783_nanogpt_model", REPO / "third_party" / "nanoGPT" / "model.py")
_nano = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_nano)

CONDITIONS = ("dense", "untied_moe", "unipool", "mirror_givens", "film_gate", "depth_embedding")


def estimate_normrouter_c(experts: int = 4, top_k: int = 1, seed: int = 78300, samples: int = 100_000) -> float:
    """Appendix-L Monte Carlo calibration for scale-stable NormRouter scores."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(samples, experts, generator=gen)
    y = F.relu(z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8))
    top = torch.topk(y, top_k, dim=-1).values
    norm = torch.linalg.vector_norm(top, dim=-1)
    positive = norm > 0
    return float(experts / top_k * torch.reciprocal(norm[positive]).mean())


class Expert(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(width, hidden)
        self.c_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class RoutedLayer(nn.Module):
    def __init__(self, config, layer_id: int, condition: str, shared_experts: bool):
        super().__init__()
        self.layer_id = layer_id
        self.condition = condition
        cfg = SimpleNamespace(n_embd=config.width, n_head=config.heads, block_size=config.block_size,
                              dropout=config.dropout, bias=True)
        self.ln_1 = _nano.LayerNorm(config.width, bias=True)
        self.attn = _nano.CausalSelfAttention(cfg)
        self.ln_2 = _nano.LayerNorm(config.width, bias=True)
        self.router = None if condition == "dense" else nn.Linear(config.width, config.experts)
        self.router_scale = nn.Parameter(torch.ones(())) if condition != "dense" else None
        self.experts = None if (condition == "dense" or shared_experts) else nn.ModuleList(
            [Expert(config.width, config.expert_hidden, config.dropout) for _ in range(config.experts)])
        self.dense_mlp = Expert(config.width, config.expert_hidden, config.dropout) if condition == "dense" else None

    def forward(self, x: torch.Tensor, shared_experts: nn.ModuleList | None,
                view_state: dict[str, torch.Tensor], config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        if self.dense_mlp is not None:
            return x + self.dense_mlp(h), h.new_zeros(()), h.new_zeros((config.experts,)), h.new_zeros(())

        router_input = h
        expert_input = h
        if self.condition == "mirror_givens":
            expert_input = apply_givens(h, view_state["givens"][self.layer_id])
        elif self.condition == "film_gate":
            group_scales = view_state["film_scale"][self.layer_id]
            group_ids = torch.arange(config.width, device=h.device) // (config.width // config.gate_groups)
            expert_input = h * group_scales[group_ids]
        elif self.condition == "depth_embedding":
            code = view_state["depth_codes"][self.layer_id]
            router_input = h + view_state["depth_projection"](code).view(1, 1, -1)
            expert_input = router_input

        logits = self.router(router_input)
        normalized = logits / torch.linalg.vector_norm(logits, dim=-1, keepdim=True).clamp_min(1e-8)
        scores = F.relu(normalized) * config.normrouter_c * self.router_scale
        top_score, top_index = scores.max(dim=-1)
        score_mass = scores.sum(dim=-1, keepdim=True)
        probs = scores / score_mass.clamp_min(1e-8)
        flat_h = expert_input.reshape(-1, expert_input.shape[-1])
        flat_idx = top_index.reshape(-1)
        flat_score = top_score.reshape(-1, 1)
        flat_out = torch.zeros_like(flat_h)
        experts = shared_experts if shared_experts is not None else self.experts
        for expert_id, expert in enumerate(experts):
            token_ids = torch.nonzero(flat_idx == expert_id, as_tuple=False).squeeze(-1)
            if token_ids.numel():
                values = expert(flat_h.index_select(0, token_ids)) * flat_score.index_select(0, token_ids)
                flat_out = flat_out.index_copy(0, token_ids, values)

        assignments = F.one_hot(top_index, num_classes=config.experts).to(probs.dtype)
        # Return per-layer sufficient terms. SmallGPT combines these into one
        # global pool-level balance loss across all layers.
        importance = scores.mean(dim=(0, 1))
        load = assignments.mean(dim=(0, 1))
        load_by_expert = load.detach()
        router_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return x + flat_out.reshape_as(x), importance, load_by_expert, router_entropy


def apply_givens(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply eight learned Givens rotations to the first sixteen channels."""
    pairs = torch.arange(16, device=x.device).reshape(8, 2)
    selected = x.index_select(-1, pairs.reshape(-1)).reshape(*x.shape[:-1], 8, 2)
    left, right = selected[..., 0], selected[..., 1]
    c, s = torch.cos(angles).view(*([1] * (x.ndim - 1)), 8), torch.sin(angles).view(*([1] * (x.ndim - 1)), 8)
    rotated = torch.stack((c * left - s * right, s * left + c * right), dim=-1).reshape(*x.shape[:-1], 16)
    out = x.clone()
    out[..., pairs.reshape(-1)] = rotated
    return out


class SmallGPT(nn.Module):
    def __init__(self, config, condition: str, vocab_size: int):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.config = config
        self.condition = condition
        self.wte = nn.Embedding(vocab_size, config.width)
        self.wpe = nn.Embedding(config.block_size, config.width)
        shared = condition in ("unipool", "mirror_givens", "film_gate", "depth_embedding")
        self.shared_experts = nn.ModuleList([Expert(config.width, config.expert_hidden, config.dropout)
                                             for _ in range(config.experts)]) if shared else None
        self.layers = nn.ModuleList([RoutedLayer(config, i, condition, shared) for i in range(config.layers)])
        self.ln_f = _nano.LayerNorm(config.width, bias=True)
        self.lm_head = nn.Linear(config.width, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.register_parameter("givens", nn.Parameter(torch.zeros(config.layers, 8)) if condition == "mirror_givens" else None)
        self.register_parameter("film_scale", nn.Parameter(torch.ones(config.layers, config.gate_groups)) if condition == "film_gate" else None)
        if condition == "depth_embedding":
            self.depth_codes = nn.Parameter(torch.zeros(config.layers, config.depth_code_dim))
            self.depth_projection = nn.Linear(config.depth_code_dim, config.width, bias=False)
        else:
            self.register_parameter("depth_codes", None)
            self.depth_projection = None
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / (2 * config.layers) ** 0.5)
        if self.givens is not None:
            nn.init.zeros_(self.givens)
        if self.film_scale is not None:
            nn.init.ones_(self.film_scale)
        if self.depth_codes is not None:
            nn.init.normal_(self.depth_codes, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        view = {"givens": self.givens, "film_scale": self.film_scale,
                "depth_codes": self.depth_codes, "depth_projection": self.depth_projection}
        importance_terms = []
        loads = []
        entropies = []
        for layer in self.layers:
            x, importance, load, entropy = layer(x, self.shared_experts, view, self.config)
            importance_terms.append(importance)
            if load.numel():
                loads.append(load)
                entropies.append(entropy)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) if targets is not None else None
        if importance_terms:
            importance_stack = torch.stack(importance_terms)
            load_stack = torch.stack(loads) if loads else torch.zeros_like(importance_stack)
            if self.condition == "untied_moe":
                per_layer = (importance_stack * load_stack).sum(dim=-1)
                aux = self.config.pool_aux_weight * self.config.experts * per_layer.mean()
            else:
                pool_importance = importance_stack.mean(dim=0)
                pool_load = load_stack.mean(dim=0)
                aux = self.config.pool_aux_weight * self.config.experts * torch.sum(pool_importance * pool_load)
        else:
            aux = logits.new_zeros(())
        load_mean = torch.stack(loads).mean(dim=0) if loads else logits.new_zeros((self.config.experts,))
        entropy_mean = torch.stack(entropies).mean() if entropies else logits.new_zeros(())
        return logits, loss, aux, load_mean, entropy_mean


class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
