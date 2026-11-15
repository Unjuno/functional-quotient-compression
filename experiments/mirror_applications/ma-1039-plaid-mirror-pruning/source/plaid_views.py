"""Small runtime adapter for viewing PLAID's shared centroid bank."""
from __future__ import annotations

import time
import types
import torch


def rotate_rows(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Apply a shared-angle block-diagonal Givens rotation to 128-D rows."""
    y = x.clone()
    a = x[..., 0::2]
    b = x[..., 1::2]
    c = torch.as_tensor(float(__import__('math').cos(angle)), dtype=x.dtype, device=x.device)
    s = torch.as_tensor(float(__import__('math').sin(angle)), dtype=x.dtype, device=x.device)
    y[..., 0::2] = c * a - s * b
    y[..., 1::2] = s * a + c * b
    return y


def install_view_adapter(ranker):
    """Keep official candidate generation/pruning; swap only its centroid score input."""
    state = {"mode": "native", "angle": 0.0, "route_seconds": 0.0,
             "candidate_pids": None, "cells": None, "centroid_scores": None}
    original_get_cells = ranker.get_cells
    original_generate_pids = ranker.generate_candidate_pids

    def get_cells_hook(self, Q, ncells):
        t0 = time.perf_counter()
        mode, angle = state["mode"], state["angle"]
        if mode == "mirror_centroid_bank":
            saved = self.codec.centroids
            try:
                self.codec.centroids = rotate_rows(saved, angle)
                cells, scores = original_get_cells(Q, ncells)
            finally:
                self.codec.centroids = saved
        elif mode == "native_query_routing":
            cells, scores = original_get_cells(rotate_rows(Q, -angle), ncells)
        else:
            cells, scores = original_get_cells(Q, ncells)
        state["route_seconds"] = time.perf_counter() - t0
        state["cells"] = cells.detach().cpu().clone()
        state["centroid_scores"] = scores.detach().cpu().clone()
        return cells, scores

    def generate_pids_hook(self, Q, ncells):
        pids, scores = original_generate_pids(Q, ncells)
        state["candidate_pids"] = torch.unique(pids.detach().cpu(), sorted=True)
        return pids, scores

    ranker.get_cells = types.MethodType(get_cells_hook, ranker)
    ranker.generate_candidate_pids = types.MethodType(generate_pids_hook, ranker)
    return state


def configure_state(state, mode, angle=0.0):
    state["mode"] = mode
    state["angle"] = float(angle)
    state["route_seconds"] = 0.0
    state["candidate_pids"] = None
    state["cells"] = None
    state["centroid_scores"] = None

