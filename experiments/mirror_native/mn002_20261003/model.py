# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Minimal shared-state Transformer. No generated expert matrices in native forward.

Column notation: FF(x) = W2 [GELU(W1 x + b1) * (1 + S.T softmax(R x + c))] + b2.
All learned arrays are dimensionless, FP32 by default. S is states x hidden,
R is states x model_dim. Soft state mixing is NOT sparse expert dispatch.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class MirrorMLP(nn.Module):
    def __init__(self, d: int, hidden: int, states: int):
        super().__init__()
        if min(d, hidden, states) < 1:
            raise ValueError('d, hidden, states must be positive')
        self.up = nn.Linear(d, hidden)
        self.down = nn.Linear(hidden, d)
        self.router = nn.Linear(d, states)
        self.states = nn.Parameter(torch.randn(states, hidden) * 1.0)

    def forward(self, x: torch.Tensor, override: str | None = None):
        p = self.router(x).softmax(-1)
        if isinstance(override, torch.Tensor):
            if override.ndim != 1 or override.numel() != p.shape[-1] or not torch.isfinite(override).all() or (override < 0).any() or not torch.isclose(override.sum(), torch.tensor(1., dtype=override.dtype)):
                raise ValueError('fixed routing must be a normalized finite probability vector')
            p = override.to(p).expand_as(p)
        elif override == 'uniform':
            p = torch.full_like(p, 1 / p.shape[-1])
        elif override == 'roll':
            p = p.roll(1, dims=-1)  # Deliberately wrong state identity, same masses.
        elif override is not None:
            raise ValueError(f'unsupported routing override: {override}')
        gain = 1 + p @ self.states
        return self.down(F.gelu(self.up(x)) * gain), p


class DenseMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.up, self.down = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x, override=None):
        return self.down(F.gelu(self.up(x))), None


class DirectGateMLP(DenseMLP):
    """Simple conditional-feature control: no state bank, gate directly from x."""
    def __init__(self, d, hidden):
        super().__init__(d, hidden)
        self.gate = nn.Linear(d, hidden)

    def forward(self, x, override=None):
        return self.down(F.gelu(self.up(x)) * (2 * self.gate(x).sigmoid())), None


class FullSoftMoE(nn.Module):
    """Larger reference, all E independent experts evaluated. Not a sparse-MoE kernel."""
    def __init__(self, d, hidden, states):
        super().__init__()
        self.experts = nn.ModuleList([DenseMLP(d, hidden) for _ in range(states)])
        self.router = nn.Linear(d, states)

    def forward(self, x, override=None):
        p = self.router(x).softmax(-1)
        if isinstance(override, torch.Tensor):
            if override.ndim != 1 or override.numel() != p.shape[-1] or not torch.isfinite(override).all() or (override < 0).any() or not torch.isclose(override.sum(), torch.tensor(1., dtype=override.dtype)):
                raise ValueError('fixed routing must be a normalized finite probability vector')
            p = override.to(p).expand_as(p)
        elif override == 'uniform':
            p = torch.full_like(p, 1 / p.shape[-1])
        elif override == 'roll':
            p = p.roll(1, dims=-1)
        elif override is not None:
            raise ValueError(f'unsupported routing override: {override}')
        ys = torch.stack([expert(x)[0] for expert in self.experts], -2)
        return (ys * p.unsqueeze(-1)).sum(-2), p


class CausalAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        if heads < 1 or d % heads:
            raise ValueError('model dimension must be divisible by positive heads')
        self.heads, self.head_dim = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [a.view(b, t, self.heads, self.head_dim).transpose(1, 2) for a in (q, k, v)]
        offset = 0
        if past is not None:
            pk, pv = past
            if pk.shape != pv.shape or pk.ndim != 4 or pk.shape[:2] != (b, self.heads) or pk.shape[-1] != self.head_dim:
                raise ValueError('incompatible KV cache shape')
            if pk.dtype != k.dtype or pk.device != k.device:
                raise ValueError('incompatible KV cache dtype/device')
            offset = pk.shape[-2]
            k, v = torch.cat((pk, k), -2), torch.cat((pv, v), -2)
        # Offset mask is required for chunks: is_causal=True alone is not enough.
        query_pos = offset + torch.arange(t, device=x.device)
        key_pos = torch.arange(k.shape[-2], device=x.device)
        mask = key_pos[None, :] <= query_pos[:, None]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        a = scores.masked_fill(~mask, float('-inf')).softmax(-1)
        out = (a @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out), (k, v)


class Block(nn.Module):
    def __init__(self, d, hidden, states, heads, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        if kind == 'mirror':
            self.ff = MirrorMLP(d, hidden, states)
        elif kind == 'dense':
            self.ff = DenseMLP(d, hidden)
        elif kind == 'direct_gate':
            self.ff = DirectGateMLP(d, hidden)
        elif kind == 'full_moe':
            self.ff = FullSoftMoE(d, hidden, states)
        else:
            raise ValueError(f'unknown MLP kind: {kind}')

    def forward(self, x, past, override):
        a, cache = self.attn(self.ln1(x), past)
        x = x + a
        y, routing = self.ff(self.ln2(x), override)
        return x + y, cache, routing


class TinyLM(nn.Module):
    def __init__(self, kind='mirror', d=32, hidden=64, states=4, layers=2,
                 heads=4, vocab=21, max_length=128):
        super().__init__()
        if min(d, hidden, states, layers, vocab, max_length) < 1:
            raise ValueError('all dimensions must be positive')
        self.config = dict(kind=kind, d=d, hidden=hidden, states=states,
                           layers=layers, heads=heads, vocab=vocab, max_length=max_length)
        self.token = nn.Embedding(vocab, d)
        self.position = nn.Embedding(max_length, d)
        self.blocks = nn.ModuleList([Block(d, hidden, states, heads, kind) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor, past=None, override=None):
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[1] == 0:
            raise ValueError('ids must be non-empty [batch, tokens] int64')
        if past is not None and len(past) != len(self.blocks):
            raise ValueError('wrong number of cached layers')
        offset = 0 if past is None else past[0][0].shape[-2]
        if offset + ids.shape[1] > self.config['max_length']:
            raise ValueError('context exceeds max_length')
        if past is not None and any(k.shape[-2] != offset for k, v in past):
            raise ValueError('inconsistent cache lengths')
        pos = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        caches, routes = [], []
        for i, layer in enumerate(self.blocks):
            x, cache, route = layer(x, None if past is None else past[i], override[i] if isinstance(override, (list, tuple)) else override)
            caches.append(cache)
            if route is not None:
                routes.append(route)
        return self.head(self.norm(x)), caches, routes
