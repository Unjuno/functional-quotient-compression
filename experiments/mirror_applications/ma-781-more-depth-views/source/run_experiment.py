"""Synthetic MoRE-like reused expert pool with heldout depth×expert roles."""
import csv, hashlib, io, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).parents[1]
SEED=78101; L=4; E=4; D=8; R=2

def pack(**arrays):
    b=io.BytesIO(); np.savez_compressed(b,**arrays); return b.getvalue()
def nrmse(pred,target): return float(np.sqrt(np.mean((pred-target)**2)/(np.mean(target**2)+1e-12)))
def run():
    rng=np.random.default_rng(SEED)
    expert=rng.normal(0,.25,(E,D,D)).astype(np.float32)
    mats=rng.normal(0,.12,(R,D,D)).astype(np.float32)
    a=rng.normal(0,.18,(L,R)).astype(np.float32); b=rng.normal(0,.18,(E,R)).astype(np.float32)
    teacher=expert[None,:,:,:]+np.einsum('lr,er,rij->leij',a,b,mats)
    held=[(i,i) for i in range(L)]; train=[(l,e) for l in range(L) for e in range(E) if (l,e) not in held]
    # Fit expert intercept and shared residual basis from training role pairs only.
    base=np.stack([np.mean([teacher[l,e] for l,e in train if e==j],axis=0) for j in range(E)]).astype(np.float32)
    residual=np.stack([teacher[l,e]-base[e] for l,e in train])
    _,_,vh=np.linalg.svd(residual.reshape(len(train),-1),full_matrices=False)
    basis=vh[:R].reshape(R,D,D).astype(np.float32)
    # Align basis signs/rotation is immaterial; project observed residuals into it.
    gram=np.einsum('rij,sij->rs',basis,basis)
    fit_start=time.perf_counter()
    coeff=np.zeros((L,E,R),np.float64)
    for l,e in train:
        rhs=np.einsum('rij,ij->r',basis,teacher[l,e]-base[e]); coeff[l,e]=np.linalg.solve(gram,rhs)
    # Fit bilinear depth×expert factors using observed role codes, initialized deterministically.
    obs=np.array(train); y=np.array([coeff[l,e] for l,e in train])
    def pred(z):
        aa=z[:L*R].reshape(L,R); bb=z[L*R:].reshape(E,R)
        return aa[obs[:,0]]*bb[obs[:,1]]
    x0=rng.normal(0,.1,L*R+E*R)
    fit=least_squares(lambda z:(pred(z)-y).ravel(),x0,max_nfev=500,xtol=1e-13,ftol=1e-13,gtol=1e-13)
    aa=fit.x[:L*R].reshape(L,R).astype(np.float32); bb=fit.x[L*R:].reshape(E,R).astype(np.float32)
    # Fit ordinary additive depth + expert code (MoRE-style learned embeddings).
    z0=rng.normal(0,.1,L*R+E*R)
    addfit=least_squares(lambda z:(z[:L*R].reshape(L,R)[obs[:,0]]+z[L*R:].reshape(E,R)[obs[:,1]]-y).ravel(),z0,max_nfev=500)
    fit_ms=(time.perf_counter()-fit_start)*1000
    da=addfit.x[:L*R].reshape(L,R).astype(np.float32); db=addfit.x[L*R:].reshape(E,R).astype(np.float32)
    methods=[]
    methods.append(('independent_layer_expert',teacher.copy(),pack(weights=teacher,role_ids=np.arange(L*E,dtype=np.int16)),D*D))
    methods.append(('hard_tied_expert_pool',np.broadcast_to(base[None],teacher.shape).copy(),pack(experts=base,expert_ids=np.arange(E,dtype=np.int16)),D*D))
    addcode=da[:,None,:]+db[None,:,:]
    addpred=base[None]+np.einsum('ler,rij->leij',addcode,basis)
    addpay=pack(experts=base,basis=basis,depth_embedding=da,expert_embedding=db,layer_ids=np.arange(L,dtype=np.int16),expert_ids=np.arange(E,dtype=np.int16))
    methods.append(('more_additive_depth_embedding',addpred,addpay,D*D+R*D*D))
    bilcode=aa[:,None,:]*bb[None,:,:]
    bilpred=base[None]+np.einsum('ler,rij->leij',bilcode,basis)
    bilpay=pack(experts=base,basis=basis,depth_factor=aa,expert_factor=bb,layer_ids=np.arange(L,dtype=np.int16),expert_ids=np.arange(E,dtype=np.int16))
    methods.append(('native_bilinear_shared_basis',bilpred,bilpay,D*D+R*D*D+R))
    methods.append(('mirror_depth_expert_views',bilpred,bilpay,D*D+R*D*D+R))
    rows=[]
    for name,predm,payload,mac in methods:
        eval_x=rng.normal(size=(64,D)).astype(np.float32)
        start=time.perf_counter()
        for _ in range(200):
            for l,e in held: _=eval_x@predm[l,e].T
        eval_ms=(time.perf_counter()-start)*1000/200
        err=nrmse(predm[[l for l,e in held],[e for l,e in held]],teacher[[l for l,e in held],[e for l,e in held]])
        rows.append({'seed':SEED,'method':name,'split':'heldout_diagonal_layer_expert_roles','serialized_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'heldout_operator_nrmse':err,'active_macs_per_role':mac,'fit_updates':0,'fit_wall_ms':round(fit_ms,6),'eval_wall_ms':round(eval_ms,6),'fit_solver_evals':fit.nfev+addfit.nfev,'status_note':'exact native bilinear alias' if name=='mirror_depth_expert_views' else ''})
    p=ROOT/'RESULTS_CORE.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows
if __name__=='__main__':
    for row in run(): print(row)
