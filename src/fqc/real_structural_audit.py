"""Task-unconditioned real-model structural audit primitives.

These routines consume correctly oriented real checkpoint weights and measure
candidate shared structure. They intentionally do not convert residuals, spans,
or concentration statistics into codec-bit or quality claims.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

from .structural_diagnostics import operator_span_spectrum


def _matrix(x) -> np.ndarray:
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2:
        raise ValueError('matrix required')
    return a


def normalized_flat_cosine(a, b) -> float:
    A=_matrix(a).reshape(-1); B=_matrix(b).reshape(-1)
    if A.shape!=B.shape:
        raise ValueError('same flattened shape required')
    den=float(np.linalg.norm(A)*np.linalg.norm(B))
    return float(A@B/den) if den>0 else 0.0


def span_summary(mats: Sequence[np.ndarray], ranks: Sequence[int]=(1,2,4)) -> dict:
    """Normalized shared-matrix-span summary.

    Each matrix is Frobenius-normalized before the family SVD so the result is
    not dominated by scale differences between family members.
    """
    s,residual=operator_span_spectrum(mats,normalize=True)
    energy=s*s
    total=float(energy.sum())
    fractions=(energy/total if total>0 else np.zeros_like(energy)).tolist()
    out={
        'member_count':len(mats),
        'singular_values':[float(x) for x in s],
        'span_energy_fractions':[float(x) for x in fractions],
        'residual_by_rank':{},
    }
    for r in sorted(set(int(x) for x in ranks if int(x)>=0)):
        if r<=len(s):
            out['residual_by_rank'][str(r)]=float(residual[r])
    return out


def split_pytorch_linear_output_heads(weight, *, heads: int, head_dim: int) -> list[np.ndarray]:
    """Split raw PyTorch Linear `[out,in]` rows into canonical `[in,head_dim]` head maps."""
    W=_matrix(weight)
    if heads<=0 or head_dim<=0 or W.shape[0]!=heads*head_dim:
        raise ValueError('weight output dimension must equal heads*head_dim')
    return [W[h*head_dim:(h+1)*head_dim,:].T.copy() for h in range(heads)]


def split_pytorch_linear_input_heads(weight, *, heads: int, head_dim: int) -> list[np.ndarray]:
    """Split raw PyTorch Linear `[out,in]` input columns into canonical `[head_dim,out]` blocks."""
    W=_matrix(weight)
    if heads<=0 or head_dim<=0 or W.shape[1]!=heads*head_dim:
        raise ValueError('weight input dimension must equal heads*head_dim')
    # Canonical row-vector operator is W.T; its input-row blocks are the raw input-column blocks transposed.
    return [W[:,h*head_dim:(h+1)*head_dim].T.copy() for h in range(heads)]


def qk_gqa_group_spans(q_weight, k_weight, *, q_heads: int, kv_heads: int, head_dim: int) -> list[dict]:
    """Shared projection-span diagnostics for each contiguous GQA Q-group plus its K head."""
    if q_heads<=0 or kv_heads<=0 or q_heads%kv_heads:
        raise ValueError('q_heads must be a positive multiple of kv_heads')
    qs=split_pytorch_linear_output_heads(q_weight,heads=q_heads,head_dim=head_dim)
    ks=split_pytorch_linear_output_heads(k_weight,heads=kv_heads,head_dim=head_dim)
    group=q_heads//kv_heads
    out=[]
    for k in range(kv_heads):
        qids=list(range(k*group,(k+1)*group))
        fam=[qs[h] for h in qids]+[ks[k]]
        out.append({'kv_head':k,'q_heads':qids,'span':span_summary(fam,ranks=(1,2,3,4))})
    return out


def value_output_operators(v_weight, o_weight, *, q_heads: int, kv_heads: int, head_dim: int) -> list[np.ndarray]:
    """Return exact per-query-head `W_V[k] @ W_O[h]` linear value-output operators.

    `v_weight` and `o_weight` are raw PyTorch `[out,in]` weights. GQA mapping is
    the standard contiguous mapping, which is separately declared by the adapter.
    """
    if q_heads<=0 or kv_heads<=0 or q_heads%kv_heads:
        raise ValueError('q_heads must be a positive multiple of kv_heads')
    vs=split_pytorch_linear_output_heads(v_weight,heads=kv_heads,head_dim=head_dim)
    os=split_pytorch_linear_input_heads(o_weight,heads=q_heads,head_dim=head_dim)
    group=q_heads//kv_heads
    return [vs[h//group]@os[h] for h in range(q_heads)]


def pairwise_cosine_matrix(mats: Sequence[np.ndarray]) -> list[list[float]]:
    xs=[_matrix(x) for x in mats]
    if not xs:
        raise ValueError('at least one matrix is required')
    shape=xs[0].shape
    if any(x.shape!=shape for x in xs):
        raise ValueError('all matrices must have identical shape')
    return [[normalized_flat_cosine(a,b) for b in xs] for a in xs]


def _top_energy_fractions(scores: np.ndarray, fractions=(0.01,0.05,0.10,0.25)) -> dict[str,float]:
    x=np.asarray(scores,dtype=np.float64).reshape(-1)
    if np.any(x<0) or not np.all(np.isfinite(x)):
        raise ValueError('scores must be finite and nonnegative')
    e=x*x; total=float(e.sum())
    order=np.sort(e)[::-1]
    out={}
    for f in fractions:
        if not (0<f<=1): raise ValueError('fractions must lie in (0,1]')
        k=max(1,int(np.ceil(f*len(order))))
        out[f'top_{100*f:g}_percent_energy_fraction']=float(order[:k].sum()/total) if total>0 else 0.0
    return out


def swiglu_channel_summary(gate_weight, up_weight, down_weight) -> dict:
    """Channel-level structural statistics for a bias-free SwiGLU MLP.

    Raw PyTorch shapes are gate/up `[intermediate,d_model]` and down
    `[d_model,intermediate]`. `||up_j||*||down_:j||` is the Frobenius norm of
    the exact outer descriptor invariant under up-branch rescaling with inverse
    down rescaling. Gate/up cosine is descriptive only.
    """
    G=_matrix(gate_weight); U=_matrix(up_weight); D=_matrix(down_weight)
    if G.shape!=U.shape or D.shape!=(G.shape[1],G.shape[0]):
        raise ValueError('incompatible SwiGLU weight shapes')
    gate_norm=np.linalg.norm(G,axis=1)
    up_norm=np.linalg.norm(U,axis=1)
    down_norm=np.linalg.norm(D,axis=0)
    descriptor_norm=up_norm*down_norm
    den=np.maximum(gate_norm*up_norm,1e-30)
    gate_up_cos=np.sum(G*U,axis=1)/den
    return {
        'channel_count':int(G.shape[0]),
        'descriptor_norm_mean':float(descriptor_norm.mean()),
        'descriptor_norm_cv':float(descriptor_norm.std()/descriptor_norm.mean()) if descriptor_norm.mean()>0 else 0.0,
        'descriptor_energy_concentration':_top_energy_fractions(descriptor_norm),
        'gate_up_cosine_mean':float(gate_up_cos.mean()),
        'gate_up_cosine_mean_abs':float(np.abs(gate_up_cos).mean()),
        'gate_up_cosine_max_abs':float(np.abs(gate_up_cos).max()),
    }


def corresponding_weight_family_summary(weights: Mapping[int,np.ndarray]) -> dict:
    """Cross-layer span/cosine summary for the same tensor role."""
    layers=sorted(weights)
    mats=[_matrix(weights[i]) for i in layers]
    return {
        'layers':layers,
        'span':span_summary(mats,ranks=(1,2,3)),
        'pairwise_cosine':pairwise_cosine_matrix(mats),
    }
