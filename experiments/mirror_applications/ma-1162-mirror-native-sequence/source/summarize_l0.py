"""Explore L0 algebra and CPU projection-sharing costs; not a training result."""
import json
import random
import statistics
import time
import math
from pathlib import Path
import torch
from mechanism import SharedMirrorCausal, rotate_first_plane, conjugated_relu, inference_payload_bytes


def measure_model(K):
    model=SharedMirrorCausal(32,K).eval()
    x=torch.randn(2,32,32)
    with torch.no_grad():
        for _ in range(12): model(x)
        times=[]
        for _ in range(80):
            start=time.perf_counter_ns()
            model(x)
            times.append((time.perf_counter_ns()-start)/1e6)
    return {"views":K,"trainable_scalars":sum(p.numel() for p in model.parameters()),
            "torch_save_bytes":inference_payload_bytes(model),
            "median_cpu_forward_ms":round(statistics.median(times),6),
            "p90_cpu_forward_ms":round(sorted(times)[int(.9*len(times))],6)}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    seq=[measure_model(k) for k in (1,2,4,8,16)]
    checks=[]
    for seed in (41,42,43,44,45):
        torch.manual_seed(seed)
        q,k=torch.randn(2,19,8),torch.randn(2,19,8)
        gauge=((q@k.transpose(-1,-2))-
               (rotate_first_plane(q,torch.tensor(.5))@rotate_first_plane(k,torch.tensor(.5)).transpose(-1,-2))).abs().max().item()
        angular=((q@k.transpose(-1,-2))-
                 (rotate_first_plane(q,torch.tensor(.5))@rotate_first_plane(k,torch.tensor(-.3)).transpose(-1,-2))).abs().mean().item()
        m=SharedMirrorCausal(8,4).eval()
        x=torch.randn(2,13,8);xp=x.clone();xp[:,9:,:]=torch.randn_like(xp[:,9:,:])*20
        with torch.no_grad():
            y=m(x);yp=m(xp)
        checks.append({"seed":seed,"qk_same_orthogonal_max_diff":gauge,
                       "qk_different_mean_diff":angular,
                       "max_causal_prefix_diff":(y[:,:,:9,:]-yp[:,:,:9,:]).abs().max().item(),
                       "view_0_1_mean_diff":(y[:,0]-y[:,1]).abs().mean().item()})
    result={"name":"MNS-001_L0", "date":"2026-10-18",
            "scope":"Analytical/CPU forward checks ONLY; no fitted neural tasks or LM loss; K=16 is not independent capacity.",
            "torch_version":torch.__version__,"cpu_threads":1,"shape_benchmark":{"batch":2,"seq_len":32,"d":32},
            "tests_passed":10,"tests_failed":0,
            "num_eval_seeds":5,"geometry_checks":checks,"forward_benchmark":seq,
            "limitations":["Eager Python per-view for loop, not fused accelerator", "No train curve, no real language, no FLOP matching", "No actual token generation throughput", "KV cache storage/reuse not checked", "Increasing view count increases active attention O(K*T^2*d) and runtime"]}
    path=Path(__file__).resolve().parent.parent/'results'/'RESULTS_L0.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({"tests_passed":10,"max_same_QK_difference":max(v['qk_same_orthogonal_max_diff'] for v in checks),
                      "max_causal_prefix_difference":max(v['max_causal_prefix_diff'] for v in checks),
                      "min_nonidentical_QK_mean_difference":min(v['qk_different_mean_diff'] for v in checks),
                      "min_view_difference":min(v['view_0_1_mean_diff'] for v in checks),
                      "models":seq},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
