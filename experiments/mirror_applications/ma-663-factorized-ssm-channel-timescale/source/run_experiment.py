import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=S=8; K=4; T=32; SEQS=16; TASKS=(16,64); RHOS=(0.,.05,.2); PRANKS=(0,2,4,8)
SCHEMA=np.frombuffer(b"CHANNEL-TIMESCALE-SSM-V1",dtype=np.uint8)

def softplus(x): return np.logaddexp(0.,x)

def make_world(seed,n,rho):
    rng=np.random.default_rng(seed*1000+n)
    uc,_=np.linalg.qr(rng.normal(size=(C,K))); vs,_=np.linalg.qr(rng.normal(size=(S,K)))
    z=rng.normal(size=(n,K)).astype(np.float32)
    cp=np.einsum('ck,nk,sk->ncs',uc,z,vs)
    bases=np.stack([np.outer(uc[:,k],vs[:,k]).reshape(-1) for k in range(K)])
    noise=rng.normal(size=(n,C*S)); noise-= (noise@bases.T)@bases
    cpflat=cp.reshape(n,-1); scale=np.sqrt(np.mean(cpflat**2,axis=1,keepdims=True))
    noise*=scale/np.maximum(np.sqrt(np.mean(noise**2,axis=1,keepdims=True)),1e-12)
    residual=rho*noise.reshape(n,C,S)
    raw=(cp+residual).astype(np.float32)
    rates=softplus(raw).astype(np.float32)
    readout=rng.normal(size=(C,S)).astype(np.float32)/np.sqrt(C*S)
    x=rng.normal(size=(n,SEQS,T,C)).astype(np.float32)
    return uc.astype(np.float32),vs.astype(np.float32),z,raw,rates,residual.astype(np.float32),readout,x

def predict(rates,readout,x):
    n,seqs,tokens,channels=x.shape
    h=np.zeros((n,seqs,C,S),dtype=np.float32); out=[]
    decay=np.exp(-rates).astype(np.float32)[:,None,:,:]
    for t in range(tokens):
        xt=x[:,:,t,:,None]
        h=decay*h+(1.-decay)*xt
        out.append(np.sum(h*readout[None,None,:,:],axis=(2,3)))
    return np.stack(out,axis=-1)

def nrmse(pred,target): return float(np.sqrt(np.mean((pred-target)**2)/max(float(np.mean(target**2)),1e-12)))

def residual_svd(residual,r):
    left=[];right=[]
    for m in residual:
        u,s,vh=np.linalg.svd(m,full_matrices=False)
        left.append((u[:,:r]*s[:r]).astype(np.float32))
        right.append(vh[:r].astype(np.float32))
    return np.asarray(left),np.asarray(right)

def save(name,payload):
    p=ROOT/'source'/name;start=time.perf_counter();np.savez_compressed(p,**payload)
    return p.stat().st_size,time.perf_counter()-start

def run_condition(seed,n,rho,prank):
    uc,vs,z,raw,rates,residual,readout,x=make_world(seed,n,rho)
    base=np.einsum('ck,nk,sk->ncs',uc,z,vs).astype(np.float32)
    if prank:
        left,right=residual_svd(residual,prank)
        private=np.einsum('ncr,nrs->ncs',left,right).astype(np.float32)
    else:
        left=np.zeros((n,C,0),np.float32);right=np.zeros((n,0,S),np.float32);private=np.zeros_like(base)
    predcp=predict(softplus(base),readout,x)
    predprivate=predict(softplus(base+private),readout,x)
    ref=predict(rates,readout,x)
    payloads={
      'independent_full':{'schema':SCHEMA,'task_ids':np.arange(n,dtype=np.uint16),'transition_rates':rates,'readout':readout},
      'native_cp':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'channel_basis':uc,'timescale_basis':vs,'task_codes':z,'readout':readout.copy()},
      'mirror_cp':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'channel_basis':uc.copy(),'timescale_basis':vs.copy(),'task_codes':z.copy(),'readout':readout.copy()},
      'native_cp_private':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'channel_basis':uc.copy(),'timescale_basis':vs.copy(),'task_codes':z.copy(),'private_left':left,'private_right':right,'readout':readout.copy()},
      'mirror_cp_private':{'schema':SCHEMA.copy(),'task_ids':np.arange(n,dtype=np.uint16),'channel_basis':uc.copy(),'timescale_basis':vs.copy(),'task_codes':z.copy(),'private_left':left.copy(),'private_right':right.copy(),'readout':readout.copy()},
    }
    predmap={'independent_full':ref,'native_cp':predcp,'mirror_cp':predcp,'native_cp_private':predprivate,'mirror_cp_private':predprivate}
    rows=[];recurrent=n*SEQS*T*C*S
    for method,payload in payloads.items():
        fname=f'{seed}_n{n}_rho{rho:g}_r{prank}_{method}.npz';size,wall=save(fname,payload)
        gen=(0 if method=='independent_full' else n*C*S*K)
        if method.endswith('_private') and prank:gen+=n*C*S*prank
        rows.append({'seed':seed,'task_count':n,'residual_rms':rho,'private_rank':prank,'method':method,'serialized_bytes':int(size),'output_nrmse':nrmse(predmap[method],ref),'sequences':n*SEQS,'tokens':n*SEQS*T,'optimizer_updates':0,'parameter_generation_mac_proxy':int(gen),'recurrent_mac_proxy':int(recurrent),'wall_time_s':wall})
    for a,b in (('native_cp','mirror_cp'),('native_cp_private','mirror_cp_private')):
        for key in payloads[a]:assert np.array_equal(payloads[a][key],payloads[b][key])
        assert np.array_equal(predmap[a],predmap[b])
    return rows

def run(seeds):
    rows=[r for seed in seeds for n in TASKS for rho in RHOS for rank in PRANKS for r in run_condition(seed,n,rho,rank)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for r in rows:w.writerow([f"n{r['task_count']}_rho{r['residual_rms']:g}_private{r['private_rank']}",r['seed'],r['method'],r['serialized_bytes'],r['tokens'],0,r['parameter_generation_mac_proxy']+r['recurrent_mac_proxy'],r['wall_time_s'],'sequence_output_nrmse',r['output_nrmse'],'parameter_generation_mac_proxy',r['parameter_generation_mac_proxy'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
