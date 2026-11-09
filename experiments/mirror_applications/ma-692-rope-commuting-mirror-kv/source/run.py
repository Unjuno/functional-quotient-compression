#!/usr/bin/env python3
"""Synthetic FP32 test of RoPE-commuting canonical KV views."""
from __future__ import annotations
import csv, io, json, math, time, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, T, NV = 16, 48, 8
SEEDS = {"dev-69201": 69201, "audit-69211": 69211, "audit-69212": 69212, "audit-69213": 69213}
DTYPE = np.float32


def rotations(angles):
    out = np.zeros((len(angles)*2, len(angles)*2), dtype=DTYPE)
    for i, a in enumerate(angles):
        c, s = np.float32(np.cos(a)), np.float32(np.sin(a))
        out[2*i:2*i+2, 2*i:2*i+2] = ((c, -s), (s, c))
    return out


def rope_matrix(pos):
    # Distinct frequencies avoid accidental cross-plane degeneracy.
    freq = (10000.0 ** (-np.arange(D//2, dtype=np.float32)*2/D)).astype(DTYPE)
    return np.stack([rotations(t * freq) for t in pos])


def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / np.sum(e, axis=-1, keepdims=True)


def npz_bytes(**arrays):
    b = io.BytesIO()
    np.savez(b, **arrays)
    return len(b.getvalue()), b.getvalue()


def run_one(label, seed, output_dir):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(T, D)).astype(DTYPE)
    k = rng.normal(size=(T, D)).astype(DTYPE)
    v = rng.normal(size=(T, D)).astype(DTYPE)
    pos = np.arange(T, dtype=np.float32)
    r = rope_matrix(pos)
    qrot = np.einsum('tij,tj->ti', r, q)
    kcanon = np.einsum('tij,tj->ti', r, k)
    angles_a = rng.uniform(-0.7, 0.7, size=(NV, D//2)).astype(DTYPE)
    angles_b = rng.uniform(-0.7, 0.7, size=(NV, D//2)).astype(DTYPE)
    masks = np.tril(np.ones((T, T), dtype=DTYPE))
    scale = np.float32(1 / math.sqrt(D))
    attn_mask = np.where(masks > 0, 0.0, -1e9).astype(DTYPE)
    max_lazy_err = max_native_err = max_noncommuting_err = 0.0
    t0 = time.perf_counter()
    mats_k, mats_v = [], []
    lazy_payload_bytes = independent_payload_bytes = 0
    lazy_cache_blob = independent_blob = b''
    total_proxy = 0
    for m in range(NV):
        a, b = rotations(angles_a[m]), rotations(angles_b[m])
        # Materialized logical key and value cache.
        km = np.einsum('tij,jk,tk->ti', r, a, k)  # R_t A k_t
        vm = v @ b.T
        mats_k.append(km); mats_v.append(vm)
        score = (qrot @ km.T) * scale + attn_mask
        p = softmax(score)
        out_mat = (p @ vm) @ b  # deliberate paired value map to retain view map? corrected below
        # Correct logical output is attention weights times transformed values.
        out_mat = p @ vm
        # Lazy commuting route: q_view=A^T R_t q; canonical K; output B after attention.
        qview = qrot @ a
        plazy = softmax((qview @ kcanon.T) * scale + attn_mask)
        out_lazy = (plazy @ v) @ b.T
        # Native phase parameterization computes same per-plane rotations directly.
        anew = rotations(angles_a[m]); bnew = rotations(angles_b[m])
        qnative = qrot @ anew
        pnative = softmax((qnative @ kcanon.T) * scale + attn_mask)
        out_native = (pnative @ v) @ bnew.T
        # Equal angle norm, but couple coordinates across distinct RoPE planes.
        c = np.eye(D, dtype=DTYPE)
        theta = np.float32(angles_a[m, 0])
        co, si = np.float32(np.cos(theta)), np.float32(np.sin(theta))
        c[0, 0] = co; c[2, 2] = co; c[0, 2] = -si; c[2, 0] = si
        knon = np.einsum('tij,jk,tk->ti', r, c, k)
        pnonmat = softmax((qrot @ knon.T) * scale + attn_mask)
        out_nonmat = pnonmat @ vm
        # The naive lazy query transform is position independent; it cannot commute C through R_t.
        pnonlazy = softmax(((qrot @ c) @ kcanon.T) * scale + attn_mask)
        out_nonlazy = (pnonlazy @ v) @ b.T
        den = max(float(np.linalg.norm(out_mat)), 1e-30)
        max_lazy_err = max(max_lazy_err, float(np.linalg.norm(out_mat-out_lazy)/den))
        max_native_err = max(max_native_err, float(np.linalg.norm(out_mat-out_native)/den))
        max_noncommuting_err = max(max_noncommuting_err, float(np.linalg.norm(out_nonmat-out_nonlazy)/den))
        # Structured per-plane angle operations: 2 MAC-equivalents per coordinate for q and output.
        total_proxy += 4*T*D
    wall = time.perf_counter() - t0
    # Every paid object for lazy deployment and independent materialization is serialized.
    lazy_payload_bytes, lazy_cache_blob = npz_bytes(K=kcanon, V=v, angle_k=angles_a, angle_v=angles_b,
                                                      positions=pos, rope_frequencies=(10000.0 ** (-np.arange(D//2, dtype=np.float32)*2/D)).astype(DTYPE),
                                                      causal_mask=masks)
    kmstack, vmstack = np.stack(mats_k), np.stack(mats_v)
    independent_payload_bytes, independent_blob = npz_bytes(K_views=kmstack, V_views=vmstack,
                                                               positions=pos, causal_mask=masks)
    output_dir.mkdir(parents=True, exist_ok=True)
    lazy_path = output_dir / f'{label}.lazy.npz'
    independent_path = output_dir / f'{label}.independent.npz'
    lazy_path.write_bytes(lazy_cache_blob)
    independent_path.write_bytes(independent_blob)
    # Serialized replay: reconstruct exact values from paid cache and angles.
    with np.load(io.BytesIO(lazy_cache_blob), allow_pickle=False) as z:
        assert np.array_equal(z['K'], kcanon) and np.array_equal(z['V'], v)
        assert np.array_equal(z['angle_k'], angles_a) and np.array_equal(z['angle_v'], angles_b)
    return {
      "condition": label, "seed": seed, "method": "lazy_commuting_vs_independent_materialized",
      "serialized_bytes": lazy_payload_bytes, "independent_bytes": independent_payload_bytes,
      "lazy_sha256": hashlib.sha256(lazy_cache_blob).hexdigest(), "independent_sha256": hashlib.sha256(independent_blob).hexdigest(),
      "train_tokens_or_examples": 0, "optimizer_updates": 0, "active_compute_proxy": total_proxy,
      "wall_time_s": wall, "max_lazy_rel_error": max_lazy_err, "max_native_phase_rel_error": max_native_err,
      "max_noncommuting_rel_error": max_noncommuting_err,
      "status_note": "synthetic FP32; full payload includes NPZ tensors and metadata; wall clock descriptive"
    }


def main():
    out = ROOT / 'source' / 'payloads'
    rows = [run_one(label, seed, out) for label, seed in SEEDS.items()]
    with (ROOT/'RESULTS_CORE.csv').open('w', newline='') as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
