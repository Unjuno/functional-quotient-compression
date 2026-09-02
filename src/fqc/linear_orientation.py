"""Checkpoint-weight orientation and canonical linear-operator conversion."""
from __future__ import annotations

from typing import Any
import numpy as np

ROW_VECTOR_X_W = 'row_vector_x_times_W'
PYTORCH_OUT_IN = 'pytorch_linear_weight_out_in'


def expected_attention_projection_shapes(
    orientation: str,
    d_model: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    value_dim: int,
) -> dict[str, list[int]]:
    if orientation == ROW_VECTOR_X_W:
        return {
            'WQ':[d_model,q_heads*head_dim],
            'WK':[d_model,kv_heads*head_dim],
            'WV':[d_model,kv_heads*value_dim],
            'WO':[q_heads*value_dim,d_model],
        }
    if orientation == PYTORCH_OUT_IN:
        return {
            'WQ':[q_heads*head_dim,d_model],
            'WK':[kv_heads*head_dim,d_model],
            'WV':[kv_heads*value_dim,d_model],
            'WO':[d_model,q_heads*value_dim],
        }
    raise ValueError(f'unsupported linear weight orientation: {orientation}')


def _numpy(value: Any) -> np.ndarray:
    x=value
    if hasattr(x,'detach'): x=x.detach()
    if hasattr(x,'cpu'): x=x.cpu()
    if hasattr(x,'numpy'): x=x.numpy()
    return np.asarray(x)


def canonical_row_matrix(weight: Any, orientation: str) -> np.ndarray:
    """Return the mathematical row-vector matrix used by FQC analysis."""
    a=_numpy(weight)
    if a.ndim!=2:
        raise ValueError('linear weight must be rank-2')
    if orientation == ROW_VECTOR_X_W:
        return a
    if orientation == PYTORCH_OUT_IN:
        return a.T
    raise ValueError(f'unsupported linear weight orientation: {orientation}')
