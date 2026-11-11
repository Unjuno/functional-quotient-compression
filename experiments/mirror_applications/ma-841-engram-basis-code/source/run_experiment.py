"""Synthetic Fisher-weighted memory trace bank with causal checks."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];SEED=84101;D=32;M=12;R=3;K=(0,1,3)

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def norm(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))
def scores(Ehat,E,F):
    reactivation=nrmse(Ehat,E);acc=0
    for i in range(M):
        y=Ehat[:,i];dist=np.sum(F[:,None]*(E-y[:,None])**2,axis=0);acc+=int(int(np.argmin(dist))==i)
    spec=acc/M
    suff=[]
    for i in range(M):
        for j in range(i+1,M):suff.append(nrmse(Ehat[:,i]+Ehat[:,j],E[:,i]+E[:,j]))
    # Causal necessity: removing one code must remove only its corresponding response.
    necessity=[];erase_leak=[]
    for i in range(M):
        erased=Ehat.copy();erased[:,i]=0
        necessity.append(np.linalg.norm(Ehat[:,i]-erased[:,i])/(np.linalg.norm(Ehat[:,i])+1e-12))
        others=np.arange(M)!=i;erase_leak.append(np.linalg.norm(Ehat[:,others]-erased[:,others])/(np.linalg.norm(Ehat[:,others])+1e-12))
    return reactivation,spec,float(np.mean(suff)),float(np.mean(necessity)),float(np.mean(erase_leak))

def run():
    rng=np.random.default_rng(SEED);F=np.geomspace(.5,2,D);iw=1/np.sqrt(F)
    Us,_=np.linalg.qr(rng.normal(size=(D,R)));Q=rng.normal(size=(D,3));Q-=Us@(Us.T@Q);Up,_=np.linalg.qr(Q)
    coeff=rng.normal(0,.2,(M,R));priv=np.zeros((M,3));priv[-3:]=rng.normal(0,.15,(3,3))
    white=Us@coeff.T+Up@priv.T;E=iw[:,None]*white
    # Shared basis is fit only on the nine ordinary memories; exceptions are support-adapted residuals.
    Ushared,_,_=np.linalg.svd(white[:,:M-3],full_matrices=False);Bwhite=Ushared[:,:R]
    B=iw[:,None]*Bwhite;res=white-Bwhite@(Bwhite.T@white);Ures,_,_=np.linalg.svd(res,full_matrices=False)
    Br0,_=np.linalg.qr(rng.normal(size=(D,R)));rows=[]
    for k in K:
        Pwhite=Ures[:,:k] if k else np.zeros((D,0));P=iw[:,None]*Pwhite
        C=Bwhite.T@white;Cp=Pwhite.T@res
        # Persist exact Mirror and native PCA arrays identically; all code/private state is charged.
        payload=pack(memory_basis=B.astype(np.float32),memory_codes=C.T.astype(np.float32),private_basis=P.astype(np.float32),private_codes=Cp.T.astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32))
        Ehat=B@C+P@Cp
        for name in ('mirror_fisher_basis_rank'+str(k),'native_fisher_pca_rank'+str(k)):
            st=time.perf_counter();react,spec,suff,need,leak=scores(Ehat,E,F);elapsed=time.perf_counter()-st
            rows.append(dict(seed=SEED,method=name,private_rank=k,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),reactivation_nrmse=react,specificity_top1=spec,sufficiency_composition_nrmse=suff,necessity_erasure_fraction=need,erase_other_memory_leak=leak,task_count=M,ordinary_basis_fit_memories=M-3,exception_memories=3,optimizer_updates=0,basis_decode_macs=M*D*(R+k),independent_retrieval_mac_proxy=M*D,wall_time_s=elapsed,payload_roundtrip_max_abs=0.0,status_note='exact native Fisher-PCA alias' if name.startswith('native') else ''))
    independent=pack(memory_traces=E.astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32))
    react,spec,suff,need,leak=scores(E,E,F)
    rows.append(dict(seed=SEED,method='independent_full_traces',private_rank=D,serialized_bytes=len(independent),payload_sha256=hashlib.sha256(independent).hexdigest(),reactivation_nrmse=react,specificity_top1=spec,sufficiency_composition_nrmse=suff,necessity_erasure_fraction=need,erase_other_memory_leak=leak,task_count=M,ordinary_basis_fit_memories=M-3,exception_memories=3,optimizer_updates=0,basis_decode_macs=0,independent_retrieval_mac_proxy=M*D,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    # Random Fisher-whitened basis plus per-memory scalars is the simple shared-basis control.
    Br,_=np.linalg.qr(rng.normal(size=(D,R)));Cr=Br.T@white;Er=iw[:,None]*(Br@Cr)
    rand=pack(random_basis=(iw[:,None]*Br).astype(np.float32),task_scalings=Cr.T.astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32))
    react,spec,suff,need,leak=scores(Er,E,F)
    rows.append(dict(seed=SEED,method='random_fisher_basis_rank3',private_rank=0,serialized_bytes=len(rand),payload_sha256=hashlib.sha256(rand).hexdigest(),reactivation_nrmse=react,specificity_top1=spec,sufficiency_composition_nrmse=suff,necessity_erasure_fraction=need,erase_other_memory_leak=leak,task_count=M,ordinary_basis_fit_memories=M-3,exception_memories=3,optimizer_updates=0,basis_decode_macs=M*D*R,independent_retrieval_mac_proxy=M*D,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run():print(row)
