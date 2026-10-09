"""Minimal CPU implementation matching EleutherAI/sparsify SparseCoder.forward."""
import torch

def forward(x: torch.Tensor, weights: dict[str, torch.Tensor], k: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = x @ weights['encoder.weight'].T + weights['encoder.bias']
    values, indices = pre.topk(k, dim=-1)
    acts = values.clamp_min(0)
    decoded = (acts.unsqueeze(-1) * weights['W_dec'][indices]).sum(dim=1) + weights['b_dec']
    skip = x @ weights['W_skip'].T
    return decoded + skip, acts, indices


def nonoverlap_starts(num_tokens: int, count: int, seed: int, window: int = 128):
    """Choose deterministic, unique aligned blocks with no token overlap."""
    import numpy as np
    n = min(count, num_tokens // window)
    blocks = np.random.default_rng(seed).choice(num_tokens // window, size=n, replace=False)
    return sorted((blocks * window).astype(int).tolist())
