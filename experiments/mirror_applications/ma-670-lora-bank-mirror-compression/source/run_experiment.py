import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=16;K=4;P=2;TARGETS=16;QUERIES=128
SOURCES=(16,64);RHOS=(0.,.05,.2)
SCHEMA=np.frombuffer(b"LORA-SOURCE-BANK-MIRROR-CODE-V1",dtype=np.uint8)

def make_world(seed,n,rho):
    rng=np.random.default_rng(seed*1000+n)
    base=(rng.normal(size=(D,D))/D).astype(np.float32)
    left=rng.normal(size=(K,D)).astype(np.float32)/np.sqrt(D)
    right=rng.normal(size=(K,D)).astype(np.float32)/np.sqrt(D)
    atoms=np.einsum('ko,ki->koi',left,right)
    source_codes=rng.normal(size=(n,K)).astype(np.float32)*.5
    shared=np.einsum('nk,koi->noi',source_codes,atoms)
    priv_left=rng.normal(size=(n,P,D)).astype(np.float32)/np.sqrt(D)
    priv_right=rng.normal(size=(n,P,D)).astype(np.float32)/np.sqrt(D)
    raw_priv=np.einsum('npo,npi->noi',priv_left,priv_right)
    scale=np.sqrt(np.mean(shared**2,axis=(1,2),keepdims=True))
    raw_scale=np.sqrt(np.mean(raw_priv**2,axis=(1,2),keepdims=True))
    priv_left*=np.sqrt(rho*scale/np.maximum(raw_scale,1e-12)).astype(np.float32)
    priv_right*=np.sqrt(rho*scale/np.maximum(raw_scale,1e-12)).astype(np.float32)
    private=np.einsum('npo,npi->noi',priv_left,priv_right).astype(np.float32)
    source=(shared+private).astype(np.float32)
    alpha=rng.normal(size=(TARGETS,n)).astype(np.float32)/np.sqrt(n)
    target_codes=alpha@source_codes
    target_priv=np.einsum('tn,npo,npi->toi',alpha,priv_left,priv_right).astype(np.float32)
    target=base[None,:,:]+np.einsum('tk,koi->toi',target_codes,atoms)+target_priv
    x=rng.normal(size=(TARGETS,QUERIES,D)).astype(np.float32)
    y=np.einsum('tqi,toi->tqo',x,target)
    return base,left,right,source_codes,priv_left,priv_right,alpha,atoms,source,target_codes,target_priv,target,x,y

def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/max(float(np.mean(b*b)),1e-12)))

def lora_factors(matrices,rank=K+P):
    aa=[];bb=[]
    for m in matrices:
        u,s,vh=np.linalg.svd(m,full_matrices=False)
        aa.append((u[:,:rank]*s[:rank]).astype(np.float32));bb.append(vh[:rank].astype(np.float32))
    return np.asarray(aa),np.asarray(bb)

def generate_targets(base,left,right,codes,pleft,pright,alpha,rho):
    atoms=np.einsum('ko,ki->koi',left,right)
    z=alpha@codes
    w=base[None,:,:]+np.einsum('tk,koi->toi',z,atoms)
    if rho>0:w+=np.einsum('tn,npo,npi->toi',alpha,pleft,pright)
    return w,z

def save(name,payload):
    p=ROOT/'source'/name;started=time.perf_counter();np.savez_compressed(p,**payload)
    return int(p.stat().st_size),time.perf_counter()-started

def run_condition(seed,n,rho):
    base,left,right,codes,pleft,pright,alpha,atoms,source,target_codes,target_priv,target,x,y=make_world(seed,n,rho)
    la,lb=lora_factors(source)
    mirror_target,generated_codes=generate_targets(base,left,right,codes,pleft,pright,alpha,rho)
    if rho==0:
        saved_alpha=np.zeros((0,n),dtype=np.float32);saved_pl=np.zeros((n,0,D),dtype=np.float32);saved_pr=np.zeros_like(saved_pl)
    else:
        saved_alpha=alpha;saved_pl=pleft;saved_pr=pright
    payloads={
      'lorahub_full_bank':{'schema':SCHEMA,'source_ids':np.arange(n,dtype=np.uint16),'target_ids':np.arange(TARGETS,dtype=np.uint16),'source_A':la,'source_B':lb,'target_coefficients':alpha,'base':base},
      'independent_full_targets':{'schema':SCHEMA.copy(),'target_ids':np.arange(TARGETS,dtype=np.uint16),'target_matrices':target},
      'mirror_shared_bank':{'schema':SCHEMA.copy(),'source_ids':np.arange(n,dtype=np.uint16),'target_ids':np.arange(TARGETS,dtype=np.uint16),'shared_left_atoms':left,'shared_right_atoms':right,'source_mirror_codes':codes,'target_mirror_codes':generated_codes,'private_left':saved_pl,'private_right':saved_pr,'target_coefficients_for_private':saved_alpha,'base':base.copy()},
      'native_vera_shared_bank':{'schema':SCHEMA.copy(),'source_ids':np.arange(n,dtype=np.uint16),'target_ids':np.arange(TARGETS,dtype=np.uint16),'shared_left':left.copy(),'shared_right':right.copy(),'source_scales':codes.copy(),'composed_scales':generated_codes.copy(),'private_left':saved_pl.copy(),'private_right':saved_pr.copy(),'target_coefficients_for_private':saved_alpha.copy(),'base':base.copy()},
    }
    generated={'lorahub_full_bank':target,'independent_full_targets':target,'mirror_shared_bank':mirror_target,'native_vera_shared_bank':mirror_target.copy()}
    rows=[];compose_full=TARGETS*n*D*D;compose_code=TARGETS*n*K+TARGETS*K*D*D
    for method,payload in payloads.items():
        size,serial_wall=save(f'{seed}_n{n}_rho{rho:g}_{method}.npz',payload)
        started=time.perf_counter();pred=np.einsum('tqi,toi->tqo',x,generated[method]);infer_wall=time.perf_counter()-started
        err=nrmse(pred,y)
        if method=='lorahub_full_bank':genmac=compose_full
        elif method=='mirror_shared_bank':genmac=compose_code+(TARGETS*n*P*D*D if rho>0 else 0)
        elif method=='native_vera_shared_bank':genmac=compose_code+(TARGETS*n*P*D*D if rho>0 else 0)
        else:genmac=0
        rows.append({'seed':seed,'source_count':n,'target_count':TARGETS,'residual_rms':rho,'method':method,'serialized_bytes':size,'target_output_nrmse':err,'queries':TARGETS*QUERIES,'optimizer_updates':0,'composition_mac_proxy':int(genmac),'inference_mac_proxy':int(TARGETS*QUERIES*D*D),'serialization_wall_time_s':serial_wall,'inference_wall_time_s':infer_wall})
    # The native VeRA-style scaling-vector representation is functionally identical.
    for a,b in [('mirror_shared_bank','native_vera_shared_bank')]:
        for u,v in [('base','base'),('shared_left_atoms','shared_left'),('shared_right_atoms','shared_right'),('source_mirror_codes','source_scales'),('target_mirror_codes','composed_scales'),('private_left','private_left'),('private_right','private_right'),('target_coefficients_for_private','target_coefficients_for_private')]:
            assert np.array_equal(payloads[a][u],payloads[b][v])
        assert np.array_equal(generated[a],generated[b])
    return rows

def run(seeds):
    rows=[r for s in seeds for n in SOURCES for rho in RHOS for r in run_condition(s,n,rho)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','inference_wall_time_s','status_note'])
        for r in rows:w.writerow([f"n{r['source_count']}_rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],r['queries'],0,r['composition_mac_proxy']+r['inference_mac_proxy'],r['serialization_wall_time_s'],'target_output_nrmse',r['target_output_nrmse'],'composition_mac_proxy',r['composition_mac_proxy'],r['inference_wall_time_s'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
