#!/usr/bin/env python3
"""Gate-3 diagnostic (no eval data): per-role cross-layer weight similarity.
Question: does any hard-sharing opportunity exist at tensor granularity?
Metric: max over layer pairs of |cosine| (sign-invariant) per role, plus
relative Frobenius distance to the role mean. Informs B-candidate design only.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "experiments/t282/code")
from engine import load_checkpoint

torch.set_num_threads(2)
cfg, state, _ = load_checkpoint("models/HF-28M", "cpu")
ROLES = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]
print(f"{'role':10s} {'max|cos|':>9s} {'pair':>9s} {'min_reldist':>11s} {'mean_reldist':>12s}")
for role in ROLES:
    tensors = []
    for i in range(cfg["num_layers"]):
        key = next(k for k in state if f".{role}.weight" in k and f".h.{i}." in k)
        a = state[key].numpy().astype(np.float64).ravel()
        tensors.append(a / np.linalg.norm(a))
    T = np.stack(tensors)
    G = T @ T.T
    iu = np.triu_indices(len(T), 1)
    amax = np.abs(G[iu]).max()
    j, k = np.unravel_index(np.argmax(np.abs(G - np.eye(len(T)))), G.shape)
    mean = T.mean(0)
    rel = [np.linalg.norm(t - mean) / np.linalg.norm(mean) for t in tensors]
    print(f"{role:10s} {amax:9.4f} {f'{j}-{k}':>9s} {min(rel):11.4f} {sum(rel)/len(rel):12.4f}")
