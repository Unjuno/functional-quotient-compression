"""Thin AdaMix/Mirror adapter wrappers around the untouched nanoGPT model."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("ma464_nanogpt_model", REPO / "third_party" / "nanoGPT" / "model.py")
_nano = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_nano)

MODES = ("single", "adamix", "mirror", "film")
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    if angles.shape[-1] != 6:
        raise ValueError("rank-four Givens view requires six angles")
    out = torch.eye(4, dtype=angles.dtype, device=angles.device)
    for index, (i, j) in enumerate(PAIR_ORDER):
        c, s = torch.cos(angles[index]), torch.sin(angles[index])
        givens = torch.eye(4, dtype=angles.dtype, device=angles.device)
        givens[i, i] = c; givens[j, j] = c
        givens[i, j] = -s; givens[j, i] = s
        out = givens @ out
    return out


class QKVAdapter(nn.Module):
    def __init__(self, base: nn.Linear, mode: str, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.base = base
        self.mode, self.rank, self.scaling = mode, rank, alpha / rank
        self.view_id = 0
        d, out = base.in_features, base.out_features
        if mode == "single":
            self.a = nn.Parameter(torch.empty(rank, d))
            self.b = nn.Parameter(torch.zeros(out, rank))
            nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        elif mode == "adamix":
            self.a = nn.Parameter(torch.empty(2, rank, d))
            self.b = nn.Parameter(torch.zeros(2, out, rank))
            for view in range(2):
                nn.init.kaiming_uniform_(self.a[view], a=5 ** 0.5)
        else:
            self.a = nn.Parameter(torch.empty(rank, d))
            self.b = nn.Parameter(torch.zeros(out, rank))
            nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
            if mode == "mirror":
                self.codes = nn.Parameter(torch.zeros(2, 6))
            else:
                self.codes = nn.Parameter(torch.ones(2, rank))
        self.route_counts = [0, 0]

    def set_view(self, view_id: int):
        if view_id not in (0, 1):
            raise ValueError(view_id)
        self.view_id = int(view_id)
        self.route_counts[self.view_id] += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.mode == "single":
            z = F.linear(x, self.a)
            return y + self.scaling * F.linear(z, self.b)
        if self.mode == "adamix":
            z = F.linear(x, self.a[self.view_id])
            return y + self.scaling * F.linear(z, self.b[self.view_id])
        z = F.linear(x, self.a)
        if self.mode == "mirror":
            r = rotation_matrix(self.codes[self.view_id])
            z = F.linear(z, r)
        else:
            z = z * self.codes[self.view_id]
        return y + self.scaling * F.linear(z, self.b)

    def view_delta(self, view_id: int) -> torch.Tensor:
        if self.mode == "single":
            return self.scaling * (self.b @ self.a)
        if self.mode == "adamix":
            return self.scaling * (self.b[view_id] @ self.a[view_id])
        if self.mode == "mirror":
            return self.scaling * (self.b @ rotation_matrix(self.codes[view_id]) @ self.a)
        return self.scaling * (self.b * self.codes[view_id].unsqueeze(0)) @ self.a

    def adapter_parameters(self):
        if self.mode == "single":
            return [self.a, self.b]
        if self.mode == "adamix":
            return [self.a, self.b]
        return [self.a, self.b, self.codes]


class AdaptedGPT(nn.Module):
    def __init__(self, base_state: dict, config, mode: str, vocab_size: int, rank: int = 4):
        super().__init__()
        self.mode = mode
        self.config = config
        self.model = _nano.GPT(config)
        self.model.load_state_dict(base_state)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # QKVAdapter modules are registered through their owning nanoGPT block;
        # this plain list is only an index, so each tensor is serialized once.
        self.adapters = []
        for block in self.model.transformer.h:
            wrapper = QKVAdapter(block.attn.c_attn, mode, rank=rank)
            block.attn.c_attn = wrapper
            self.adapters.append(wrapper)
        self.set_view(0, count=False)

    def set_view(self, view_id: int, count: bool = True):
        for adapter in self.adapters:
            adapter.view_id = int(view_id)
            if count:
                adapter.route_counts[view_id] += 1

    def forward(self, idx, targets=None):
        return self.model(idx, targets)

    def adapter_parameters(self):
        return [p for adapter in self.adapters for p in adapter.adapter_parameters()]

    def view_deltas(self, view_id: int) -> list[torch.Tensor]:
        return [adapter.view_delta(view_id) for adapter in self.adapters]

    def merged_model(self, views: tuple[int, ...] | None = None):
        views = views or ((0,) if self.mode == "single" else (0, 1))
        merged = _nano.GPT(self.config)
        wrapped = self.model.state_dict()
        plain_state = {}
        for key, value in wrapped.items():
            if ".attn.c_attn.base." in key:
                plain_state[key.replace(".attn.c_attn.base.", ".attn.c_attn.")] = value
            elif ".attn.c_attn." in key and (key.endswith(".a") or key.endswith(".b") or key.endswith(".codes")):
                continue
            else:
                plain_state[key] = value
        merged.load_state_dict(plain_state)
        with torch.no_grad():
            for layer_id, adapter in enumerate(self.adapters):
                delta = torch.stack([adapter.view_delta(v) for v in views]).mean(0)
                merged.transformer.h[layer_id].attn.c_attn.weight.add_(delta)
        return merged

    def bank_adapter_state(self):
        state = {}
        for layer, adapter in enumerate(self.adapters):
            if self.mode == "adamix":
                state[f"layer{layer}.A"] = adapter.a.detach().cpu().contiguous()
                state[f"layer{layer}.B"] = adapter.b.detach().cpu().contiguous()
            elif self.mode == "single":
                state[f"layer{layer}.A"] = adapter.a.detach().cpu().contiguous()
                state[f"layer{layer}.B"] = adapter.b.detach().cpu().contiguous()
            else:
                state[f"layer{layer}.A"] = adapter.a.detach().cpu().contiguous()
                state[f"layer{layer}.B"] = adapter.b.detach().cpu().contiguous()
                state[f"layer{layer}.codes"] = adapter.codes.detach().cpu().contiguous()
        return state


def make_gpt_config(vocab_size: int, block_size: int = 128):
    return _nano.GPTConfig(block_size=block_size, vocab_size=vocab_size,
                           n_layer=4, n_head=4, n_embd=64, dropout=0.0, bias=True)
