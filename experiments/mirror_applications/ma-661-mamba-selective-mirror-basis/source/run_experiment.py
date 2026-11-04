import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=H=8; T=32; NSEQ=128; RANKS=(1,2,4,8); RHOS=(0.,.1,.3)
SCHEMA=np.frombuffer(b"SELECTIVE-DIAGONAL-SSM-V1",dtype=np.uint8)

def softplus(x):
    return np.logaddexp(0.,x)

def make_world(seed,rho):
    rng=np.random.default_rng(seed*1000+int(rho*1000))
    v,_=np.linalg.qr(rng.normal(size=(D,4)))
    v=v.T
    weights=[]
    for _ in range(3):
        u=rng.normal(size=(H,4))/2
        w=u@v
        e=rng.normal(size=(H,D)); e-= (e@v.T)@v
        w+=rho*e
        weights.append(w.astype(np.float32))
    a_log=np.log(rng.uniform(.08,.8,size=H)).astype(np.float32)
    delta_bias=np.full(H,.4,dtype=np.float32)
    x=rng.normal(size=(NSEQ,T,D)).astype(np.float32)
    return weights,a_log,delta_bias,x

def forward(weights,a_log,delta_bias,x):
    wb,wc,wd=weights; rate=np.exp(a_log)
    h=np.zeros((len(x),H),dtype=np.float32); ys=[]
    for t in range(T):
        xt=x[:,t]
        b=xt@wb.T; c=xt@wc.T; d=softplus(xt@wd.T+delta_bias)
        decay=np.exp(-d*rate)
        h=decay*h+(1.-decay)*b*xt[:,0:1]
        ys.append(np.sum(c*h,axis=1,keepdims=True)/np.sqrt(H))
    return np.concatenate(ys,axis=1)

def nrmse(pred,target):
    return float(np.sqrt(np.mean((pred-target)**2)/max(np.mean(target**2),1e-12)))

def factor_independent(weights,k):
    out=[];pred=[]
    for w in weights:
        u,s,vh=np.linalg.svd(w,full_matrices=False); left=u[:,:k]*s[:k]; right=vh[:k]
        out.append((left.astype(np.float32),right.astype(np.float32)))
        pred.append((left@right).astype(np.float32))
    return out,pred

def factor_joint(weights,k):
    joined=np.concatenate(weights,axis=0)
    _,_,vh=np.linalg.svd(joined,full_matrices=False); v=vh[:k]
    us=[(w@v.T).astype(np.float32) for w in weights]
    return v.astype(np.float32),us,[(u@v).astype(np.float32) for u in us]

def save(name,payload):
    p=ROOT/'source'/name;t=time.perf_counter();np.savez_compressed(p,**payload)
    return int(p.stat().st_size),time.perf_counter()-t

def run_condition(seed,rho,k):
    weights,a_log,db,x=make_world(seed,rho); yref=forward(weights,a_log,db,x)
    ind,pind=factor_independent(weights,k);v,us,pjoint=factor_joint(weights,k)
    payloads={
      'mamba_full':{'schema':SCHEMA,'A_log':a_log,'delta_bias':db,'W_B':weights[0],'W_C':weights[1],'W_Delta':weights[2]},
      'native_independent_lowrank':{'schema':SCHEMA.copy(),'A_log':a_log.copy(),'delta_bias':db.copy(),'left_B':ind[0][0],'right_B':ind[0][1],'left_C':ind[1][0],'right_C':ind[1][1],'left_Delta':ind[2][0],'right_Delta':ind[2][1]},
      'native_joint_basis':{'schema':SCHEMA.copy(),'A_log':a_log.copy(),'delta_bias':db.copy(),'input_basis':v,'U_B':us[0],'U_C':us[1],'U_Delta':us[2]},
      'mirror_shared_view':{'schema':SCHEMA.copy(),'A_log':a_log.copy(),'delta_bias':db.copy(),'input_basis':v.copy(),'U_B':us[0].copy(),'U_C':us[1].copy(),'U_Delta':us[2].copy()},
    }
    preds={'mamba_full':weights,'native_independent_lowrank':pind,'native_joint_basis':pjoint,'mirror_shared_view':pjoint}
    rows=[]
    recurrent=NSEQ*T*2*H
    for method,payload in payloads.items():
        size,wall=save(f'{seed}_rho{rho:g}_k{k}_{method}.npz',payload)
        pred=forward(preds[method],a_log,db,x)
        if method=='mamba_full': gen=3*H*D
        elif method=='native_independent_lowrank': gen=3*(H*k+k*D)
        else: gen=D*k+3*H*k
        rows.append({'seed':seed,'residual_rms':rho,'rank':k,'method':method,'serialized_bytes':size,'output_nrmse':nrmse(pred,yref),'sequence_count':NSEQ,'tokens':NSEQ*T,'optimizer_updates':0,'generator_mac_proxy':int(NSEQ*T*gen),'recurrent_mac_proxy':int(recurrent),'wall_time_s':wall,'offline_svd_count':int(method in ('native_independent_lowrank','native_joint_basis','mirror_shared_view'))})
    # Matched native joint-basis and Mirror encode the exact same arrays/functions.
    assert all(np.array_equal(payloads['native_joint_basis'][x],payloads['mirror_shared_view'][x]) for x in payloads['native_joint_basis'])
    assert np.array_equal(preds['native_joint_basis'][0],preds['mirror_shared_view'][0])
    return rows

def run(seeds):
    rows=[r for s in seeds for rho in RHOS for k in RANKS for r in run_condition(s,rho,k)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
        for r in rows:w.writerow([f"rho{r['residual_rms']:g}_k{r['rank']}",r['seed'],r['method'],r['serialized_bytes'],r['tokens'],0,r['generator_mac_proxy']+r['recurrent_mac_proxy'],r['wall_time_s'],'sequence_output_nrmse',r['output_nrmse'],'generator_mac_proxy',r['generator_mac_proxy'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
