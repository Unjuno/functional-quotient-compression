"""Gauge-invariant screening diagnostics for a bank of LoRA deltas.

This is an ORACLE screening harness: projectors are fitted to training-task
weight updates, and evaluation uses held-out update weights if provided.
Results are evidence about subspace representability, NOT a trained Mirror
model, functional capacity, downstream quality or real serialized bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def canonical_delta_svd(B: np.ndarray, A: np.ndarray, rtol: float = 1e-10):
    """Return SVD of D=B@A without materializing D; invariant to GL(r) gauge.

    U, s, V have shapes (d_out,r_eff), (r_eff,), (d_in,r_eff).
    Individual signed singular vectors are not unique for equal singular
    values; their projectors and nonzero singular values are invariant.
    """
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
        raise ValueError('B must be [d_out,r], A must be [r,d_in]')
    qB, rB = np.linalg.qr(B, mode='reduced')
    qA, rA = np.linalg.qr(A.T, mode='reduced')
    uc, s, vht = np.linalg.svd(rB @ rA.T, full_matrices=False)
    keep = s > (max(s[0], 1.0) * rtol) if len(s) else np.zeros(0, dtype=bool)
    return qB @ uc[:, keep], s[keep], qA @ vht.T[:, keep]


def projector(U: np.ndarray) -> np.ndarray:
    """Gauge-invariant orthogonal projector onto the given column subspace."""
    U = np.asarray(U, dtype=np.float64)
    return U @ U.T


def principal_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Cosines of principal angles for orthonormal column bases."""
    if U.shape[0] != V.shape[0]:
        raise ValueError('basis ambient dimensions do not match')
    if U.shape[1] == 0 or V.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.svd(U.T @ V, compute_uv=False)


def fit_shared_uv(deltas: Sequence[np.ndarray], k: int):
    """Fit shared output/input spaces on TRAIN tasks only.

    Uses a two-sided concat SVD as a shared-core baseline, not Mirror-
    specific learning. Never include audit deltas in this function.
    """
    if not deltas or k <= 0:
        raise ValueError('nonempty training deltas and positive k required')
    mats = [np.asarray(x, dtype=np.float64) for x in deltas]
    shape = mats[0].shape
    if any(x.ndim != 2 or x.shape != shape for x in mats):
        raise ValueError('all trained deltas must have identical shapes')
    if k > min(shape):
        raise ValueError('k exceeds matrix minimum dimension')
    U, _, _ = np.linalg.svd(np.concatenate(mats, axis=1), full_matrices=False)
    _, _, Vh = np.linalg.svd(np.concatenate(mats, axis=0), full_matrices=False)
    return U[:, :k], Vh[:k, :].T


def projection_error(D: np.ndarray, U: np.ndarray, V: np.ndarray, mode='full', sparse_q=0):
    """Relative Frobenius error for oracle task code projected on fixed U,V.

    full: dense kxk per-task core; diagonal: k per-task coefficients;
    sparse: top-q per-task core entries (an oracle code with index cost).
    """
    D = np.asarray(D, dtype=np.float64)
    if U.shape[0] != D.shape[0] or V.shape[0] != D.shape[1]:
        raise ValueError('shared bases mismatch delta shape')
    C = U.T @ D @ V
    if mode == 'diagonal':
        C = np.diag(np.diag(C))
    elif mode == 'sparse':
        if sparse_q < 0 or sparse_q > C.size:
            raise ValueError('sparse_q outside core size')
        Z = np.zeros_like(C)
        if sparse_q:
            idx = np.argpartition(np.abs(C).ravel(), -sparse_q)[-sparse_q:]
            Z.flat[idx] = C.flat[idx]
        C = Z
    elif mode != 'full':
        raise ValueError('unknown code mode')
    residual = D - U @ C @ V.T
    return float(np.linalg.norm(residual, 'fro') / max(np.linalg.norm(D, 'fro'), 1e-30))


def gauge_diagnostic(B: np.ndarray, A: np.ndarray, G: np.ndarray):
    """Worst gauge-invariant update/projector/singular-spectrum deviations."""
    G = np.asarray(G, dtype=np.float64)
    if G.shape != (B.shape[1], B.shape[1]):
        raise ValueError('G must be invertible rank-by-rank')
    A2 = np.linalg.solve(G, A)
    B2 = B @ G
    U1, s1, V1 = canonical_delta_svd(B, A)
    U2, s2, V2 = canonical_delta_svd(B2, A2)
    return {
        'delta_max_abs': float(np.max(np.abs(B @ A - B2 @ A2))),
        'singular_max_abs': float(np.max(np.abs(s1 - s2))) if len(s1) == len(s2) and len(s1) else 0.0,
        'column_projector_max_abs': float(np.max(np.abs(projector(U1) - projector(U2)))),
        'row_projector_max_abs': float(np.max(np.abs(projector(V1) - projector(V2)))),
    }


def screen_manifest(manifest: Path, k: int, sparse_q: int):
    """Read explicitly matched-base trained LoRA factors from local .npz files.

    Manifest: {"base_revision":"...", "layer":"q_proj",
      "items":[{"task":"...", "split":"train|audit", "file":"x.npz",
                "base_revision":"..."}, ...]}
    Each .npz must hold B[d_out,r] and A[r,d_in]. Train and audit task
    IDs must be disjoint; exact model revision and layer required.
    """
    obj = json.loads(manifest.read_text(encoding='utf-8'))
    base = obj.get('base_revision')
    layer = obj.get('layer')
    if not isinstance(base, str) or not base or not isinstance(layer, str) or not layer:
        raise ValueError('explicit nonempty base_revision and layer required')
    tasks = set()
    train, audit = [], []
    for item in obj['items']:
        if item.get('base_revision') != base:
            raise ValueError('base revision mismatch; deltas are not comparable')
        task = item['task']
        if task in tasks:
            raise ValueError('duplicate task identity across train/audit')
        tasks.add(task)
        f = (manifest.parent / item['file']).resolve()
        with np.load(f, allow_pickle=False) as dat:
            B, A = dat['B'], dat['A']
        D = np.asarray(B, dtype=np.float64) @ np.asarray(A, dtype=np.float64)
        if item['split'] == 'train':
            train.append((task, D))
        elif item['split'] == 'audit':
            audit.append((task, D))
        else:
            raise ValueError('invalid split; only train/audit')
    if not train or not audit:
        raise ValueError('need disjoint nonempty train and audit task sets')
    U, V = fit_shared_uv([D for _, D in train], k)
    outcomes = []
    for task, D in audit:
        outcomes.append({
            'task':task,
            'full_core_error':projection_error(D, U, V, 'full'),
            'diagonal_error':projection_error(D, U, V, 'diagonal'),
            'sparse_core_error':projection_error(D, U, V, 'sparse', sparse_q=sparse_q),
        })
    return {'scope':'oracle weight-space representability only; not model task quality',
            'base_revision':base,'layer':layer,'train_tasks':len(train),'audit_tasks':len(audit),
            'shared_rank':k,'sparse_q':sparse_q,'results':outcomes,
            'storage_warning':'Count basis, metadata, private residuals, all adapted layers and real serializer. This tool does not measure deployed bytes.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--rank',required=True,type=int)
    p.add_argument('--sparse-q',type=int,default=0)
    args = p.parse_args()
    print(json.dumps(screen_manifest(args.manifest,args.rank,args.sparse_q),sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
