"""CPU-only small multihead exact-folding benchmark; MHA vs canonical cache.

Each method returns SAME final output for each static angle pair, and times
only decode. Native hard-shared (without View functions) is not equivalent.
"""
import argparse
import csv
import math
import statistics
import time
from pathlib import Path
import torch
from multihead import SharedMultihead, rotate_pairs, attention, code_pairwise_kl


def bench_one(B,T,H,repeats=31,warmups=7):
    torch.manual_seed(2000+T*31+H*53+B*7)
    dmod,dh=16,8
    m=SharedMultihead(d_model=dmod,d_head=dh,heads=H,mode='linear').eval()
    with torch.no_grad():
        m.q_code.copy_(torch.linspace(-.95,.85,H))
        m.v_code.copy_(torch.linspace(.75,-.65,H))
        x=torch.randn((B,T,dmod))
        last=x[:,-1:,:]
        q0=last@m.Wq
        k0=x@m.Wk
        v0=x@m.Wv
        # independent resident K/V copies, with View-specific Value and
        # physically copied Key tensors (even though no k right rotation).
        ks=[k0.clone() for _ in range(H)]
        vs=[rotate_pairs(v0,m.v_code[j]).contiguous() for j in range(H)]
        pmat=[rotate_pairs(torch.eye(dh),m.q_code[j]) for j in range(H)]
        wq_heads=[m.Wq@p for p in pmat]
        readout=torch.randn((H*dh,dmod))/math.sqrt(H*dh)
        readout_blocks=[readout[j*dh:(j+1)*dh,:] for j in range(H)]
        readout_native=[pmat_j@wb for pmat_j,wb in zip([rotate_pairs(torch.eye(dh),m.v_code[j]) for j in range(H)],readout_blocks)]

        def materialized():
            pieces=[]
            for j in range(H):
                q=rotate_pairs(q0,m.q_code[j])
                o=attention(q,ks[j],vs[j])
                pieces.append(o)
            return torch.cat(pieces,dim=-1)@readout

        def canonical_lazy():
            return m.decode_lazy(q0,k0,v0).flatten(-2)@readout

        def static_folded_native():
            y=last.new_zeros((B,1,dmod))
            for j in range(H):
                q=last@wq_heads[j]
                o=attention(q,k0,v0)
                y+=o@readout_native[j]
            return y

        def hard_shared_native():
            o=attention(q0,k0,v0)
            return torch.cat([o for _ in range(H)],dim=-1)@readout

        funcs={'materialized_cache':materialized,'canonical_lazy':canonical_lazy,
               'folded_native':static_folded_native,'hard_shared_native':hard_shared_native}
        ys={name:f() for name,f in funcs.items()}
        max_err=max((ys['materialized_cache']-ys[name]).abs().max().item() for name in ('canonical_lazy','folded_native'))
        hard_err=(ys['materialized_cache']-ys['hard_shared_native']).abs().max().item()
        # Per-head cache size counts the two physically resident K,V tensors.
        cached_bytes=2*B*T*dh*4
        model_bytes=m.serialized_bytes()
        results=[]
        for name,func in funcs.items():
            for _ in range(warmups):func()
            durations=[]
            for _ in range(repeats):
                a=time.perf_counter_ns();func();durations.append((time.perf_counter_ns()-a)*1e-6)
            ds=sorted(durations)
            n=len(ds)
            results.append({'batch':B,'prefix':T,'logical_heads':H,'method':name,
                'median_ms':statistics.median(ds),'p90_ms':ds[math.ceil(.9*n)-1],
                'max_abs_vs_materialized':max_err if name!='hard_shared_native' else hard_err,
                'canonical_kv_bytes':cached_bytes,
                'resident_kv_bytes':cached_bytes*H if name=='materialized_cache' else cached_bytes,
                'model_state_bytes':model_bytes,
                'folded_q_projection_extra_bytes':0 if name!='folded_native' else H*dmod*dh*4,
                'dtype':'fp32','torch_threads':torch.get_num_threads(),
                'repeats':repeats,'warmups':warmups})
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--repeats',type=int,default=31)
    p.add_argument('--warmups',type=int,default=7)
    a=p.parse_args()
    torch.set_num_threads(1)
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for B in (1,4):
      for H in (1,4,8):
       for T in (128,512,2048,8192):
        new=bench_one(B,T,H,a.repeats,a.warmups)
        rows+=new
        print(f'B={B} T={T} H={H} '+ ' '.join(f'{z["method"]}={z["median_ms"]:.3f}ms' for z in new),flush=True)
        with out.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('rows',len(rows))

if __name__=='__main__':main()
