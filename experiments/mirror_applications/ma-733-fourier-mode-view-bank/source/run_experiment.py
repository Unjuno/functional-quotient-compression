#!/usr/bin/env python3
"""Synthetic spectral operator bank screen for MA-733."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 73301
MODES, DOMAINS, FIT = 64, 16, 12
BANDS = ((0, 8), (8, 20), (20, 40), (40, 64))
RHOS = (0.0, 0.05, 0.2)
TOPKS = (0, 2, 4, 8, 16)
RESOLUTIONS = (128, 256)
FIELDS = 128


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    return buff.getvalue()


def _make_problem():
    rng = np.random.default_rng(SEED)
    k = np.arange(1, MODES + 1, dtype=np.float64)
    base = np.exp(-0.00058 * k * k)
    atoms = np.zeros((4, MODES), dtype=np.float64)
    for b, (lo, hi) in enumerate(BANDS):
        atoms[b, lo:hi] = 0.20 + 0.06 * np.cos(np.linspace(0, np.pi, hi-lo))
    # The address is known at deployment; held-out addresses are unseen in basis fitting.
    codes = rng.uniform(-0.7, 0.7, size=(DOMAINS, 4))
    codes[:, 0] += np.linspace(-0.2, 0.2, DOMAINS)
    structured = base[None, :] + codes @ atoms
    noise = rng.normal(size=(DOMAINS, MODES))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    spectra = np.stack([structured + rho * noise for rho in RHOS])
    return rng, base, atoms, codes, spectra


def _topk_residual(target, reconstruction, k, rng=None):
    residual = target - reconstruction
    if k == 0:
        return np.zeros_like(residual), np.empty(0, np.int16), np.empty(0, np.float32)
    if rng is None:
        idx = np.argpartition(np.abs(residual), -k)[-k:]
    else:
        idx = rng.choice(len(residual), size=k, replace=False)
    vals = residual[idx].astype(np.float32)
    out = np.zeros_like(residual)
    out[idx] = vals
    return out, idx.astype(np.int16), vals


def _metric(rng, target, approx, resolution):
    # Smooth random real fields; evaluate only the retained positive-frequency band.
    fields = rng.normal(size=(FIELDS, resolution // 2 + 1))
    freq = np.arange(fields.shape[1])
    fields /= (1.0 + freq[None, :] ** 1.5)
    x = fields[:, 1:MODES+1]
    ref = x * target[None, :]
    start = time.perf_counter()
    got = x * approx[None, :]
    elapsed = (time.perf_counter()-start)*1000
    err = np.linalg.norm(got-ref) / max(np.linalg.norm(ref), 1e-30)
    hi = slice(32, MODES)
    hi_err = np.linalg.norm(got[:, hi]-ref[:, hi]) / max(np.linalg.norm(ref[:, hi]), 1e-30)
    return float(err), float(hi_err), elapsed


def run():
    rng, base, atoms, codes, spectra = _make_problem()
    rows = []
    for ri, rho in enumerate(RHOS):
        targets = spectra[ri]
        # Fit four native spectral atoms on the 12 training domain identities only.
        centered = targets[:FIT] - base[None, :]
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        pca_atoms = vt[:4]
        pca_codes = (targets - base[None, :]) @ pca_atoms.T
        pca_recon = base[None, :] + pca_codes @ pca_atoms
        band_codes = (targets - base[None, :]) @ atoms.T / (np.sum(atoms*atoms, axis=1)[None, :] + 1e-30)
        band_recon = base[None, :] + band_codes @ atoms
        for topk in TOPKS:
            for residual_kind in (("magnitude", "random") if topk else ("none",)):
                approx_by_method = {}
                resid_idx, resid_val = [], []
                for method, reconstruction, basis, method_codes in (
                    ("native_pca_shared", pca_recon, pca_atoms, pca_codes),
                    ("native_shared_band", band_recon, atoms, band_codes),
                    ("mirror_band_view", band_recon, atoms, band_codes),
                ):
                    approx = reconstruction.copy()
                    local_idx, local_val = [], []
                    for d in range(DOMAINS):
                        if residual_kind == "none":
                            idx = np.empty(0, np.int16); val = np.empty(0, np.float32)
                        else:
                            rrng = np.random.default_rng(SEED + 1000*ri + 31*d + topk)
                            if residual_kind == "random":
                                _, idx, val = _topk_residual(targets[d], reconstruction[d], topk, rrng)
                            else:
                                _, idx, val = _topk_residual(targets[d], reconstruction[d], topk)
                            approx[d, idx] += val
                        local_idx.append(idx); local_val.append(val)
                    packed_idx = np.concatenate(local_idx).astype(np.int16) if local_idx else np.empty(0,np.int16)
                    packed_val = np.concatenate(local_val).astype(np.float32) if local_val else np.empty(0,np.float32)
                    offsets = np.cumsum([0] + [len(x) for x in local_idx]).astype(np.int32)
                    method_payload = _npz_bytes(base=base.astype(np.float32), basis=basis.astype(np.float32),
                        codes=method_codes.astype(np.float32), residual_offsets=offsets,
                        residual_indices=packed_idx, residual_values=packed_val,
                        domain_ids=np.arange(DOMAINS, dtype=np.int16), rho=np.asarray([rho],np.float32))
                    sha = hashlib.sha256(method_payload).hexdigest()
                    approx_by_method[method] = (approx, method_payload, sha)
                    resid_idx.extend(local_idx); resid_val.extend(local_val)
                for method, (approx_method, method_payload, sha) in approx_by_method.items():
                    for resolution in RESOLUTIONS:
                        errs=[]; hierrs=[]; elapsed=[]
                        for d in range(DOMAINS):
                            field_rng = np.random.default_rng(SEED + resolution*100 + d)
                            e, he, ms = _metric(field_rng, targets[d], approx_method[d], resolution)
                            errs.append(e); hierrs.append(he); elapsed.append(ms)
                        rows.append(dict(seed=SEED,rho=rho,method=method,private_topk=topk,
                            residual_selection=residual_kind,resolution=resolution,
                            serialized_bytes=len(method_payload),payload_sha256=sha,
                            relative_error=float(np.mean(errs)),high_band_error=float(np.mean(hierrs)),
                            active_multiplies=DOMAINS*MODES*FIELDS,wall_ms=float(np.sum(elapsed)),status_note="exact native alias"))
        # Independent per-domain spectra upper control.
        payload=_npz_bytes(spectra=targets.astype(np.float32), domain_ids=np.arange(DOMAINS,dtype=np.int16))
        for resolution in RESOLUTIONS:
            rows.append(dict(seed=SEED,rho=rho,method="independent_full",private_topk=MODES,
                residual_selection="full",resolution=resolution,serialized_bytes=len(payload),
                payload_sha256=hashlib.sha256(payload).hexdigest(),relative_error=0.0,
                high_band_error=0.0,active_multiplies=DOMAINS*MODES*FIELDS,wall_ms=0.0,
                status_note="reference"))
    out=ROOT/"RESULTS_CORE.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"rows":len(rows),"output":str(out),"min_bytes":min(r['serialized_bytes'] for r in rows)},indent=2))


if __name__ == "__main__":
    run()
