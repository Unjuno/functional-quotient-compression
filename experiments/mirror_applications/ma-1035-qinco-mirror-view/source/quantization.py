# Copyright (c) Meta Platforms, Inc. and affiliates.
# The pairwise assignment operations below follow QINCo v1's `utils.py`.
# Official upstream: https://github.com/facebookresearch/Qinco
# Commit: 5a324954d5c9b3700d4407d6cc24c3db6e52890e (CC-BY-NC).
import torch


def pairwise_distances(a, b):
    return (a.square().sum(-1)[:, None] + b.square().sum(-1)[None, :] - 2 * a @ b.T)


def compute_batch_distances(a, b):
    return a.square().sum(-1).unsqueeze(-1) + b.square().sum(-1).unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))


def assign_batch_multiple(x, zqs):
    bs, _, d = zqs.shape
    idx = compute_batch_distances(x.unsqueeze(1), zqs).squeeze(1).argmin(1)
    return idx, zqs[torch.arange(bs, device=x.device), idx]


def assign_to_codebook(x, c, bs=16384):
    if len(x) * len(c) < bs * bs:
        return pairwise_distances(x, c).argmin(1)
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    cn = c.square().sum(1)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        dist = xb.square().sum(1, keepdim=True) + cn[None, :] - 2 * xb @ c.T
        out[i:i + len(xb)] = dist.argmin(1)
    return out
