import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=4; L=128; SEQS=8; TASKS=(16,64); RANKS=(4,8,16); RHOS=(0.,.05,.2); TRUE_RANK=8
NFFT=256; SCHEMA=np.frombuffer(b"IMPLICIT-CAUSAL-FILTER-CODE-V1",dtype=np.uint8)

def dct_basis(k):
    t=np.arange(L,dtype=np.float64)[None,:];f=np.arange(k,dtype=np.float64)[:,None]
    b=np.cos(np.pi/L*(t+.5)*f)*np.sqrt(2./L);b[0]*=1/np.sqrt(2.)
    return b.astype(np.float32)

def make_world(seed,n,rho):
    rng=np.random.default_rng(seed*1000+n)
    basis=dct_basis(TRUE_RANK)
    codes=rng.normal(size=(n,C,TRUE_RANK)).astype(np.float32)*.5
    shared=np.einsum('nck,kl->ncl',codes,basis)
    residual=rng.normal(size=(n,C,L));residual-=np.einsum('ncl,kl->nck',residual,basis)@basis
    scale=np.sqrt(np.mean(shared**2,axis=2,keepdims=True))
    residual*=scale/np.maximum(np.sqrt(np.mean(residual**2,axis=2,keepdims=True)),1e-12)
    kernels=(shared+rho*residual).astype(np.float32)
    x=rng.normal(size=(n,SEQS,C,L)).astype(np.float32)
    return kernels,x

def causal_fft_conv(x,kernels):
    xf=np.fft.rfft(x,n=NFFT,axis=-1)
    kf=np.fft.rfft(kernels[:,None,:,:],n=NFFT,axis=-1)
    y=np.fft.irfft(xf*kf,n=NFFT,axis=-1)[...,:L]
    return y.astype(np.float32)

def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/max(float(np.mean(b*b)),1e-12)))

def learned_pca(kernels,k):
    bases=[];codes=[];pred=[]
    for c in range(C):
        _,_,vh=np.linalg.svd(kernels[:,c,:],full_matrices=False)
        b=vh[:k].astype(np.float32);code=(kernels[:,c,:]@b.T).astype(np.float32)
        bases.append(b);codes.append(code);pred.append(code@b)
    return np.asarray(bases),np.stack(codes,axis=1),np.stack(pred,axis=1)

def save(name,payload):
    p=ROOT/'source'/name;started=time.perf_counter();np.savez_compressed(p,**payload)
    return int(p.stat().st_size),time.perf_counter()-started

def run_condition(seed,n,rho,k):
    kernels,x=make_world(seed,n,rho);yref=causal_fft_conv(x,kernels)
    dct=dct_basis(k);dct_codes=np.einsum('ncl,kl->nck',kernels,dct)
    dct_pred=np.einsum('nck,kl->ncl',dct_codes,dct)
    basis,codes,pca_pred=learned_pca(kernels,k)
    payloads={
      'independent_full':{'schema':SCHEMA,'filter_ids':np.arange(n,dtype=np.uint16),'kernels':kernels},
      'native_dct_code':{'schema':SCHEMA.copy(),'filter_ids':np.arange(n,dtype=np.uint16),'frequency_ids':np.arange(k,dtype=np.uint8),'codes':dct_codes.astype(np.float32)},
      'mirror_pca_code':{'schema':SCHEMA.copy(),'filter_ids':np.arange(n,dtype=np.uint16),'filter_basis':basis,'codes':codes},
      'native_pca_code':{'schema':SCHEMA.copy(),'filter_ids':np.arange(n,dtype=np.uint16),'filter_basis':basis.copy(),'codes':codes.copy()},
    }
    preds={'independent_full':kernels,'native_dct_code':dct_pred,'mirror_pca_code':pca_pred,'native_pca_code':pca_pred}
    rows=[];fft_mac=n*SEQS*C*NFFT*np.log2(NFFT)
    for method,payload in payloads.items():
        size,wall=save(f'{seed}_n{n}_rho{rho:g}_k{k}_{method}.npz',payload)
        gen=0 if method=='independent_full' else (n*C*L*k)
        started=time.perf_counter();y=causal_fft_conv(x,preds[method]);infer_wall=time.perf_counter()-started
        rows.append({'seed':seed,'filter_count':n,'residual_rms':rho,'rank':k,'method':method,'serialized_bytes':size,'output_nrmse':nrmse(y,yref),'sequences':n*SEQS,'tokens':n*SEQS*L,'optimizer_updates':0,'filter_generation_mac_proxy':int(gen),'fft_mac_proxy':int(fft_mac),'inference_wall_time_s':infer_wall,'serialization_wall_time_s':wall})
    assert all(np.array_equal(payloads['mirror_pca_code'][k0],payloads['native_pca_code'][k0]) for k0 in payloads['mirror_pca_code'])
    assert np.array_equal(preds['mirror_pca_code'],preds['native_pca_code'])
    return rows

def run(seeds):
    rows=[r for s in seeds for n in TASKS for rho in RHOS for k in RANKS for r in run_condition(s,n,rho,k)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
        for r in rows:w.writerow([f"n{r['filter_count']}_rho{r['residual_rms']:g}_k{r['rank']}",r['seed'],r['method'],r['serialized_bytes'],r['tokens'],0,r['filter_generation_mac_proxy']+r['fft_mac_proxy'],r['inference_wall_time_s'],'causal_output_nrmse',r['output_nrmse'],'fft_mac_proxy',r['fft_mac_proxy'],r['serialization_wall_time_s'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
