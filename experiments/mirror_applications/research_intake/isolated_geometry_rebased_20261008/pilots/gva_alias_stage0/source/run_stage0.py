#!/usr/bin/env python3
"""GVA readout-only physical cache alias + cost pilot. No LLM training.

Protocol frozen in GitHub before five fresh seeds.
CPU torch float32, threads=1; no hidden calibration/audit access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

BATCH = 1
HEADS = 4
DV = 16
DK = 16
DP = 8
T = 128
ROLES = 8
CODE_DIM = 3
DEV = [11, 12, 13]
FRESH = [101, 102, 103, 104, 105]
WARMUP = 20
MEASURE = 100


def max_abs(x, y) -> float:
    return float((x - y).abs().max().item())


def gva_forward(q: torch.Tensor, qp: torch.Tensor, v: torch.Tensor, kp: torch.Tensor,
                M: torch.Tensor, explicit: bool = False):
    """One grouped-V GVA read with decoupled positional channel.

    q: B,H,DK; qp: B,H,DP; v: B,T,DV; kp: B,T,DP; M: DV,DK.
    """
    if explicit:
        kcontent = v @ M  # an extra physical content key materialization (reference only)
        sc = torch.einsum("bhd,btd->bht", q, kcontent)
    else:
        qa = q @ M.T  # absorb projection to query side
        sc = torch.einsum("bhd,btd->bht", qa, v)
    sp = torch.einsum("bhp,btp->bht", qp, kp)
    score = (sc + sp) / math.sqrt(DK)
    weights = torch.softmax(score, dim=-1)
    out = torch.einsum("bht,btd->bhd", weights, v)
    return score, out


def npz_nbytes(**arrays) -> int:
    buf = io.BytesIO()
    np.savez(buf, **{k: v.detach().numpy() if torch.is_tensor(v) else v for k,v in arrays.items()})
    return len(buf.getvalue())


def storage_bytes(*arrs) -> int:
    seen = {}
    for t in arrs:
        assert torch.is_tensor(t)
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
    return sum(seen.values())


def timings(run, reps=MEASURE) -> tuple[float, float]:
    for _ in range(WARMUP):
        run()
    ns = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        run()
        ns.append(time.perf_counter_ns() - t0)
    return (float(np.quantile(ns, 0.50)) / 1e6, float(np.quantile(ns, 0.95)) / 1e6)


def world(seed: int) -> dict:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=rng, dtype=torch.float32)
    v = rand(BATCH, T, DV)*0.35
    kpos = rand(BATCH, T, DP)*0.30
    q = rand(BATCH, HEADS, DK)*0.25
    qp = rand(BATCH, HEADS, DP)*0.25
    m0 = rand(DV, DK)*0.40
    basis = rand(CODE_DIM, DV, DK)*0.20
    codes = rand(ROLES, CODE_DIM)*0.50
    maps = m0[None, :, :] + torch.einsum("rk,kij->rij", codes, basis)

    # Same cached storage, differing ONLY the readout role m. The native baseline
    # also reuses these two tensors without allocation or recomputation.
    role_ptrs = [(v.data_ptr(), kpos.data_ptr()) for _ in range(ROLES)]
    native_single_prefix_bytes = storage_bytes(v, kpos)
    clones = [(v.clone(), kpos.clone()) for _ in range(ROLES)]
    native_per_role_copy_bytes = storage_bytes(*[a for pair in clones for a in pair])
    alias_all = len(set(role_ptrs)) == 1 and native_single_prefix_bytes*(ROLES) == native_per_role_copy_bytes

    worst_score_error = 0.0
    worst_output_error = 0.0
    for role in range(ROLES):
        s_native, o_native = gva_forward(q, qp, v, kpos, maps[role], explicit=True)
        s_shared, o_shared = gva_forward(q, qp, v, kpos, maps[role], explicit=False)
        reconstructed = m0 + torch.einsum("k,kij->ij", codes[role], basis)
        s_mirror, o_mirror = gva_forward(q, qp, v, kpos, reconstructed, explicit=False)
        worst_score_error = max(worst_score_error, max_abs(s_native, s_shared), max_abs(s_native, s_mirror))
        worst_output_error = max(worst_output_error, max_abs(o_native, o_shared), max_abs(o_native, o_mirror))

    # IMPORTANT UNSAFE negative control: a different role changes upstream
    # activations at the prefix. Reusing the original prefix is now invalid.
    mix = rand(DV,DV) / math.sqrt(DV)
    v_new = v + 0.4*torch.tanh(v @ mix)
    kp_new = kpos + 0.4*torch.tanh(kpos)
    _, output_target = gva_forward(q,qp,v_new,kp_new,maps[1],False)
    _, output_invalid = gva_forward(q,qp,v,kpos,maps[1],False)
    unsafe_output_error = max_abs(output_target, output_invalid)

    th = 0.63
    rot = torch.tensor([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]], dtype=torch.float64)
    valid = torch.tensor([[1.2,-0.25],[0.25,1.2]], dtype=torch.float64)
    invalid = torch.tensor([[1.,.35],[0.,1.]], dtype=torch.float64)
    valid_commutator_error = max_abs(rot @ valid, valid @ rot)
    invalid_commutator_error = max_abs(rot @ invalid, invalid @ rot)

    # Native matrix storage, shared basis+code (Mirror), identical native
    # shared-basis code (the closest matched control).
    native_maps_bytes = npz_nbytes(M_roles=maps)
    common = dict(M_shared=m0, B=basis, code=codes)
    mirror_maps_bytes = npz_nbytes(**common)
    native_factorized_bytes = npz_nbytes(**common)
    # Real cache allocation does NOT vary between Mirror and single-prefix native GVA.
    native_latency = timings(lambda: gva_forward(q,qp,v,kpos,maps[1],explicit=False))
    mirror_latency = timings(lambda: gva_forward(q,qp,v,kpos,
                                 m0 + torch.einsum("k,kij->ij", codes[1],basis),explicit=False))

    return dict(seed=seed,score_error=worst_score_error,output_error=worst_output_error,
      cache_alias=int(alias_all),cache_bytes=native_single_prefix_bytes,
      cache_bytes_cloned=native_per_role_copy_bytes,cache_ratio=native_single_prefix_bytes/native_per_role_copy_bytes,
      unsafe_output_error=unsafe_output_error,
      rope_valid_commutator_error=valid_commutator_error,
      rope_invalid_commutator_error=invalid_commutator_error,
      native_full_maps_npz_bytes=native_maps_bytes,mirror_maps_npz_bytes=mirror_maps_bytes,
      native_linear_code_npz_bytes=native_factorized_bytes,
      map_ratio=mirror_maps_bytes/native_maps_bytes,
      native_time_p50_ms=native_latency[0],native_time_p95_ms=native_latency[1],
      mirror_time_p50_ms=mirror_latency[0],mirror_time_p95_ms=mirror_latency[1],
      p95_ratio=mirror_latency[1]/native_latency[1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["dev","fresh"],required=True)
    parser.add_argument("--output",default="/mnt/data/mirror_gva_stage0")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    seeds=DEV if args.phase=="dev" else FRESH
    os.makedirs(args.output,exist_ok=True)
    rows=[world(seed) for seed in seeds]
    path=Path(args.output)/f"{args.phase}_raw.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    flags={
      "score_and_output_exact":all(r["score_error"]<=3e-5 and r["output_error"]<=3e-5 for r in rows),
      "physical_cache_alias":all(r["cache_alias"] for r in rows),
      "upstream_change_detected":all(r["unsafe_output_error"]>1e-4 for r in rows),
      "valid_rope_commutes":all(r["rope_valid_commutator_error"]<=1e-12 for r in rows),
      "invalid_rope_rejected":all(r["rope_invalid_commutator_error"]>1e-4 for r in rows),
      "factorized_baseline_parity":all(r["mirror_maps_npz_bytes"]==r["native_linear_code_npz_bytes"] for r in rows),
      "mirror_beats_native_factorized_bytes_10pct":all(r["mirror_maps_npz_bytes"]<=0.90*r["native_linear_code_npz_bytes"] for r in rows),
    }
    system={"python":platform.python_version(),"python_platform":platform.platform(),"cpu_logical":os.cpu_count(),
      "torch":torch.__version__,"numpy":np.__version__,"torch_threads":torch.get_num_threads(),
      "torch_dtype":"float32", "device":"cpu", "batch":BATCH,"context":T,"head_count":HEADS,
      "cuda_available":torch.cuda.is_available(),"clock":"time.perf_counter_ns", "warmup":WARMUP,"timed_calls":MEASURE}
    summary={"phase":args.phase,"seeds":seeds,"flags":flags,"system":system,
      "mean_score_error":statistics.mean(r["score_error"] for r in rows),
      "max_output_error":max(r["output_error"] for r in rows),
      "min_unsafe_error":min(r["unsafe_output_error"] for r in rows),
      "median_cache_ratio":statistics.median(r["cache_ratio"] for r in rows),
      "median_map_ratio":statistics.median(r["map_ratio"] for r in rows),
      "median_p95_ratio":statistics.median(r["p95_ratio"] for r in rows),
      "measurements":len(rows),
      "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "raw_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    summary_path=Path(args.output)/f"{args.phase}_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__":main()
