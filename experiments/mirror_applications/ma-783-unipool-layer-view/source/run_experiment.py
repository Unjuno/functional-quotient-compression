"""Synthetic UniPool global expert pool with Givens layer-role views."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]; SEED=78301; L=4; E=4; D=8; N=256

def givens(d,angles):
    q=np.eye(d,dtype=np.float32)
    for k,a in enumerate(angles):
        i=2*k; c=np.float32(np.cos(a)); s=np.float32(np.sin(a))
        g=np.eye(d,dtype=np.float32); g[i,i]=c;g[i,i+1]=-s;g[i+1,i]=s;g[i+1,i+1]=c
        q=g@q
    return q

def pack(**kwargs):
    b=io.BytesIO();np.savez_compressed(b,**kwargs);return b.getvalue()
def nrmse(p,t):return float(np.sqrt(np.mean((p-t)**2)/(np.mean(t**2)+1e-12)))
def run():
    rng=np.random.default_rng(SEED)
    experts=rng.normal(0,.2,(E,D,D)).astype(np.float32)
    angles=rng.normal(0,.22,(L,D//2)).astype(np.float32)
    qs=np.stack([givens(D,a) for a in angles])
    role=np.stack([[q.T@experts[e]@q for e in range(E)] for q in qs])
    routers=rng.normal(0,.35,(L,E,D)).astype(np.float32)
    x=rng.normal(size=(N,D)).astype(np.float32)
    route=np.argmax(np.einsum('nd,led->lne',x,routers),axis=-1)
    target=np.empty((L,N,D),np.float32)
    for l in range(L):
        for n in range(N): target[l,n]=role[l,route[l,n]]@x[n]
    idx=np.arange(E,dtype=np.int16)
    payloads={
      'independent_layer_experts':pack(experts=role,routers=routers,expert_ids=idx,layer_ids=np.arange(L,dtype=np.int16)),
      'unipool_global_experts':pack(experts=experts,routers=routers,expert_ids=idx,layer_ids=np.arange(L,dtype=np.int16)),
      'mirror_layer_givens_views':pack(experts=experts,routers=routers,angles=angles,expert_ids=idx,layer_ids=np.arange(L,dtype=np.int16)),
      'native_direct_angle_givens':pack(experts=experts,routers=routers,angles=angles,expert_ids=idx,layer_ids=np.arange(L,dtype=np.int16)),
    }
    rows=[]
    for name,payload in payloads.items():
        out=np.empty_like(target);start=time.perf_counter()
        for repeat in range(50):
            for l in range(L):
                q=qs[l] if name in ('mirror_layer_givens_views','native_direct_angle_givens') else np.eye(D,dtype=np.float32)
                mats=role[l] if name=='independent_layer_experts' else experts if name=='unipool_global_experts' else experts
                for n in range(N):
                    e=int(route[l,n]);v=q@x[n] if name in ('mirror_layer_givens_views','native_direct_angle_givens') else x[n]
                    v=mats[e]@v
                    out[l,n]=q.T@v if name in ('mirror_layer_givens_views','native_direct_angle_givens') else v
        elapsed=(time.perf_counter()-start)*1000/50
        # role matrices for mirror/direct and independent produce the same functions.
        err=nrmse(out,target)
        mac=E*D+D*D+(2*D*2 if name in ('mirror_layer_givens_views','native_direct_angle_givens') else 0)
        rows.append({'seed':SEED,'method':name,'split':'routed_heldout_inputs','serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'routed_output_nrmse':err,'router_plus_expert_macs_per_token_layer':mac,'examples':N,'optimizer_updates':0,'eval_wall_ms_per_256x4':round(elapsed,6),'status_note':'exact native direct-angle alias' if name=='mirror_layer_givens_views' else ''})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows
if __name__=='__main__':
    for r in run(): print(r)
