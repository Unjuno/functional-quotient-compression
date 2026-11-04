import argparse,csv,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=16;LAYERS=2;E=8;K=4;P=2;N=256;RHOS=(0.,.05,.2)
SCHEMA=np.frombuffer(b"MOLE-MIRROR-CODE-GATE-V1",dtype=np.uint8)

def softmax(x):
    z=x-np.max(x,axis=-1,keepdims=True);v=np.exp(z);return v/np.sum(v,axis=-1,keepdims=True)

def make_world(seed,rho):
    rng=np.random.default_rng(seed*1000+int(rho*1000))
    base=(rng.normal(size=(LAYERS,D,D))/np.sqrt(D)).astype(np.float32)*.25
    left=rng.normal(size=(LAYERS,K,D)).astype(np.float32)/np.sqrt(D)
    right=rng.normal(size=(LAYERS,K,D)).astype(np.float32)/np.sqrt(D)
    atoms=np.einsum('lko,lki->lkoi',left,right)
    codes=rng.normal(size=(LAYERS,E,K)).astype(np.float32)*.4
    shared=np.einsum('lek,lkoi->leoi',codes,atoms)
    prleft=rng.normal(size=(LAYERS,E,P,D)).astype(np.float32)/np.sqrt(D)
    prright=rng.normal(size=(LAYERS,E,P,D)).astype(np.float32)/np.sqrt(D)
    residual=np.einsum('lepo,lepi->leoi',prleft,prright)
    scales=np.sqrt(np.mean(shared**2,axis=(2,3),keepdims=True))
    rscale=np.sqrt(np.mean(residual**2,axis=(2,3),keepdims=True))
    factor=np.sqrt(rho*scales/np.maximum(rscale,1e-12)).astype(np.float32)
    prleft*=factor;prright*=factor
    residual=np.einsum('lepo,lepi->leoi',prleft,prright).astype(np.float32)
    expert=shared+residual
    router=rng.normal(size=(LAYERS,E,D)).astype(np.float32)/np.sqrt(D)
    x=rng.normal(size=(N,D)).astype(np.float32)
    return base,left,right,codes,prleft,prright,expert,router,x

def lora_factors(expert):
    aa=np.zeros((LAYERS,E,D,K+P),np.float32);bb=np.zeros((LAYERS,E,K+P,D),np.float32)
    for l in range(LAYERS):
        for e in range(E):
            u,s,vh=np.linalg.svd(expert[l,e],full_matrices=False)
            aa[l,e]=u[:,:K+P]*s[:K+P];bb[l,e]=vh[:K+P]
    return aa,bb

def gate(x,router,topk=None):
    g=softmax(x@router.T)
    if topk:
        idx=np.argpartition(g,-topk,axis=1)[:,-topk:]
        mask=np.zeros_like(g);np.put_along_axis(mask,idx,1.,axis=1);g*=mask;g/=np.maximum(g.sum(axis=1,keepdims=True),1e-12)
    return g

def forward_full(x,base,expert,router,topk=None):
    h=x
    for l in range(LAYERS):
        g=gate(h,router[l],topk)
        ex=np.einsum('nd,eod->neo',h,expert[l])
        h=h@base[l].T+np.einsum('ne,neo->no',g,ex)
        if l==0:h=np.maximum(h,0.)
    return h

def forward_shared(x,base,left,right,codes,router,prleft=None,prright=None,topk=None):
    h=x;atoms=np.einsum('lko,lki->lkoi',left,right)
    for l in range(LAYERS):
        g=gate(h,router[l],topk)
        mix=np.einsum('ne,ek->nk',g,codes[l])
        atom_out=np.einsum('nd,kod->nko',h,atoms[l])
        update=np.einsum('nk,nko->no',mix,atom_out)
        if prleft is not None and prleft.shape[2]:
            z=np.einsum('nd,epd->nep',h,prright[l])
            residual=np.einsum('nep,epo->neo',z,prleft[l])
            update+=np.einsum('ne,neo->no',g,residual)
        h=h@base[l].T+update
        if l==0:h=np.maximum(h,0.)
    return h

def mac_proxy(method,rho):
    base=N*LAYERS*D*D;gate_mac=N*LAYERS*D*E
    if method=='mole_full_bank':active=N*LAYERS*E*D*D
    elif method=='mole_full_top2':active=N*LAYERS*2*D*D
    else:
        active=N*LAYERS*K*D*D+N*LAYERS*E*K
        if method.endswith('_private') and rho>0:active+=N*LAYERS*E*P*2*D
    return int(base+gate_mac+active)

def save(name,payload):
    p=ROOT/'source'/name;start=time.perf_counter();np.savez_compressed(p,**payload)
    return int(p.stat().st_size),time.perf_counter()-start

def run_condition(seed,rho):
    base,left,right,codes,prleft,prright,expert,router,x=make_world(seed,rho)
    target=forward_full(x,base,expert,router)
    aa,bb=lora_factors(expert)
    reconstructed=np.einsum('leor,leri->leoi',aa,bb)
    paid_prleft=prleft if rho>0 else None
    paid_prright=prright if rho>0 else None
    private_payload={} if rho==0 else {'private_left':prleft,'private_right':prright}
    native_private_payload={} if rho==0 else {'private_left':prleft.copy(),'private_right':prright.copy()}
    payloads={
      'mole_full_bank':{'schema':SCHEMA,'base':base,'router':router,'expert_A':aa,'expert_B':bb},
      'mole_full_top2':{'schema':SCHEMA.copy(),'base':base.copy(),'router':router.copy(),'expert_A':aa.copy(),'expert_B':bb.copy(),'top_k':np.asarray([2],np.uint8)},
      'mirror_code':{'schema':SCHEMA.copy(),'base':base.copy(),'router':router.copy(),'atom_left':left,'atom_right':right,'expert_codes':codes},
      'native_shared_basis':{'schema':SCHEMA.copy(),'base':base.copy(),'router':router.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'expert_scales':codes.copy()},
      'mirror_code_private':{'schema':SCHEMA.copy(),'base':base.copy(),'router':router.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'expert_codes':codes.copy(),**private_payload},
      'native_shared_basis_private':{'schema':SCHEMA.copy(),'base':base.copy(),'router':router.copy(),'atom_left':left.copy(),'atom_right':right.copy(),'expert_scales':codes.copy(),**native_private_payload},
    }
    methods={
      'mole_full_bank':lambda:forward_full(x,base,reconstructed,router),
      'mole_full_top2':lambda:forward_full(x,base,reconstructed,router,topk=2),
      'mirror_code':lambda:forward_shared(x,base,left,right,codes,router),
      'native_shared_basis':lambda:forward_shared(x,base,left,right,codes,router),
      'mirror_code_private':lambda:forward_shared(x,base,left,right,codes,router,paid_prleft,paid_prright),
      'native_shared_basis_private':lambda:forward_shared(x,base,left,right,codes,router,paid_prleft,paid_prright),
    }
    rows=[]
    for method,payload in payloads.items():
        size,serial=save(f'{seed}_rho{rho:g}_{method}.npz',payload)
        start=time.perf_counter();pred=methods[method]();wall=time.perf_counter()-start
        rows.append({'seed':seed,'residual_rms':rho,'method':method,'serialized_bytes':size,'output_nrmse':float(np.sqrt(np.mean((pred-target)**2)/max(float(np.mean(target**2)),1e-12))),'examples':N,'optimizer_updates':0,'active_mac_proxy':mac_proxy(method,rho),'inference_wall_time_s':wall,'serialization_wall_time_s':serial})
    for a,b in [('mirror_code','native_shared_basis'),('mirror_code_private','native_shared_basis_private')]:
        for key_a,key_b in [('base','base'),('router','router'),('atom_left','atom_left'),('atom_right','atom_right'),('expert_codes' if a=='mirror_code' else 'expert_codes','expert_scales' if b=='native_shared_basis' else 'expert_scales'),('private_left','private_left'),('private_right','private_right')]:
            if key_a not in payloads[a]:continue
            assert np.array_equal(payloads[a][key_a],payloads[b][key_b])
        assert np.array_equal(methods[a](),methods[b]())
    return rows

def run(seeds):
    rows=[r for seed in seeds for rho in RHOS for r in run_condition(seed,rho)]
    (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
        for r in rows:w.writerow([f"rho{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'development; fresh sealed'])
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
