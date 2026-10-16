"""Exploratory computational diagnostic, NOT a trained model or production kernel benchmark.
Shared causal token SDPA + same-time low-rank view-axis SDPA vs repeated/joint SDPA.
"""
import csv
import statistics
import time
from pathlib import Path
import torch
from torch.nn import functional as F

B, D, R, K = 2, 64, 16, 4
BASE=Path(__file__).parent
torch.set_num_threads(1)
torch.manual_seed(64017)
U = torch.linalg.qr(torch.randn(D,R))[0].contiguous()
codes = torch.randn(K,R).mul_(0.25).add_(1.0)
scales = torch.randn(K,D).mul_(0.15).add_(1.0)

def base(x):
    return F.scaled_dot_product_attention(x[:,None,:,:], x[:,None,:,:], x[:,None,:,:], is_causal=True)[:,0]

def repeated(x):
    z = (x[:,None,:,:] * scales[None,:,None,:]).reshape(B*K,1,len(x[0]),D)
    out = F.scaled_dot_product_attention(z,z,z,is_causal=True)
    return out.reshape(B,K,len(x[0]),D).transpose(1,2)

def factorized(x):
    b=base(x)
    z=(b @ U)[:,:,None,:]*codes[None,None,:,:]
    # View-axis attention, independently per token, never sees future tokens.
    z2 = z.reshape(B*x.shape[1],1,K,R)
    w=F.scaled_dot_product_attention(z2,z2,z2, is_causal=False).reshape(B,x.shape[1],K,R)
    return b[:,:,None,:] + (w @ U.T)

def full_joint(x):
    t=x.shape[1]
    z=(x[:,:,None,:]*scales[None,None,:,:]).reshape(B,t*K,D)
    pos=torch.arange(t).repeat_interleave(K)
    mask=(pos[:,None]>=pos[None,:])
    return F.scaled_dot_product_attention(z[:,None,:,:],z[:,None,:,:],z[:,None,:,:],attn_mask=mask)[:,0].reshape(B,t,K,D)

def benchmark(func,x, warm=20, count=35, groups=4):
    med=[]
    for _ in range(groups):
        with torch.no_grad():
            for _ in range(warm): func(x)
            times=[]
            for _ in range(count):
                st=time.perf_counter_ns();func(x);times.append((time.perf_counter_ns()-st)/1e6)
        med.append(statistics.median(times))
    return statistics.median(med),min(med),max(med)

def main():
    rows=[]
    for t in (128,256,512):
        x=torch.randn(B,t,D)
        # compare causal dependency (future change cannot change first time output)
        y=factorized(x)
        x2=x.clone();x2[:,t//2+1:]+=13
        y2=factorized(x2)
        diff=(y[:,:t//2+1]-y2[:,:t//2+1]).abs().max().item()
        assert diff < 1e-6, diff
        # Confirm nonzero gradients wrt a view-specific code, not a dead branch.
        c=codes.clone().requires_grad_(True)
        bs=base(x)
        z=(bs @ U)[:,:,None,:]*c[None,None,:,:]
        zz=z.reshape(B*t,1,K,R)
        mix=F.scaled_dot_product_attention(zz,zz,zz).reshape(B,t,K,R)
        loss=(mix[:,:,0,:]**2).mean()
        grad=torch.autograd.grad(loss,c)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum().item()>1e-8
        for name,method in (("single_token",base),( "4_independent_token",repeated),("shared_token_plus_view_attention",factorized),("joint_T_times_K_attention",full_joint)):
            med,lo,hi=benchmark(method,x)
            rows.append({"T":t,"B":B,"d":D,"K":K,"r":R,"method":name,"median_ms":round(med,5),"range_low_ms":round(lo,5),"range_high_ms":round(hi,5),"causal_test_max_diff":diff})
            print(t,name,round(med,4),'ms')
    path=BASE/'benchmarks.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print('pilot diagnostics PASSED; wrote',path)
if __name__=='__main__':main()
