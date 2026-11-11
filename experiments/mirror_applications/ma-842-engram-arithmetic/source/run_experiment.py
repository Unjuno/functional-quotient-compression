"""Synthetic simultaneous engram composition through Fisher-weighted codes."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=84201;D=32;M=20;R=4;K=(0,1,3);NQ=256

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def evalbank(Vhat,V,F,Q):
    pred=Vhat@Q;truth=V@Q;mix=nrmse(pred,truth);single=nrmse(Vhat,V)
    # Top-1 specificity on one-memory cues under the Fisher metric.
    hits=0
    for i in range(M):
        dist=np.sum(F[:,None]*(V-Vhat[:,i,None])**2,axis=0);hits+=int(np.argmin(dist)==i)
    # Signed cancellation and single-code erasure tests.
    cancel=max(float(np.linalg.norm(Vhat[:,i]-Vhat[:,i])) for i in range(M))
    need=[];leak=[]
    for i in range(M):
        erased=Vhat.copy();erased[:,i]=0
        need.append(np.linalg.norm(Vhat[:,i]-erased[:,i])/(np.linalg.norm(Vhat[:,i])+1e-12))
        keep=np.arange(M)!=i;leak.append(np.linalg.norm(erased[:,keep]-Vhat[:,keep])/(np.linalg.norm(Vhat[:,keep])+1e-12))
    return mix,single,hits/M,cancel,float(np.mean(need)),float(np.mean(leak))

def run():
    rng=np.random.default_rng(SEED);F=np.geomspace(.5,2,D);iw=1/np.sqrt(F)
    U,_=np.linalg.qr(rng.normal(size=(D,R)));Qp=rng.normal(size=(D,3));Qp-=U@(U.T@Qp);Up,_=np.linalg.qr(Qp)
    C=rng.normal(0,.2,(R,M));Cp=np.zeros((3,M));Cp[:,-3:]=rng.normal(0,.15,(3,3))
    V=iw[:,None]*(U@C+Up@Cp);white=V/iw[:,None]
    # Fit shared memory basis on 17 ordinary traces only.
    Ub,_,_=np.linalg.svd(white[:,:M-3],full_matrices=False);Bwhite=Ub[:,:R];B=iw[:,None]*Bwhite
    residual=white-Bwhite@(Bwhite.T@white);Ur,_,_=np.linalg.svd(residual,full_matrices=False)
    Q=np.zeros((M,NQ),np.float32);avg=[]
    for q in range(NQ):
        n=int(rng.integers(2,9));ids=rng.choice(M,n,replace=False);Q[ids,q]=rng.choice([-1,1],size=n);avg.append(n)
    onehots=np.eye(M,dtype=np.float32);Qall=np.concatenate([Q,onehots],axis=1)
    rows=[]
    for k in K:
        Pwhite=Ur[:,:k] if k else np.zeros((D,0));P=iw[:,None]*Pwhite
        Cm=Bwhite.T@white;Cpm=Pwhite.T@residual;Vhat=B@Cm+P@Cpm
        payload=pack(memory_basis=B.astype(np.float32),memory_codes=Cm.T.astype(np.float32),private_basis=P.astype(np.float32),private_codes=Cpm.T.astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32))
        for name in ('mirror_code_superposition_rank'+str(k),'native_pca_code_addition_rank'+str(k)):
            st=time.perf_counter();mix,single,spec,cancel,need,leak=evalbank(Vhat,V,F,Qall);elapsed=time.perf_counter()-st
            active=float(np.mean(np.count_nonzero(Q,axis=0)))
            rows.append(dict(seed=SEED,method=name,private_rank=k,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),mixture_composition_nrmse=mix,single_memory_reactivation_nrmse=single,specificity_top1=spec,signed_cancellation_error=cancel,necessity_erasure_fraction=need,erasure_other_memory_leak=leak,tested_mixtures=NQ,mean_mixture_size=active,memory_count=M,optimizer_updates=0,code_sum_and_decode_macs=NQ*(active*(R+k)+D*(R+k)),raw_vector_add_macs=NQ*active*D,wall_time_s=elapsed,payload_roundtrip_max_abs=0.0,status_note='exact native factorization/addition alias' if name.startswith('native') else ''))
    indep=pack(memory_traces=V.astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32));mix,single,spec,cancel,need,leak=evalbank(V,V,F,Qall)
    rows.append(dict(seed=SEED,method='independent_raw_addition',private_rank=D,serialized_bytes=len(indep),payload_sha256=hashlib.sha256(indep).hexdigest(),mixture_composition_nrmse=mix,single_memory_reactivation_nrmse=single,specificity_top1=spec,signed_cancellation_error=cancel,necessity_erasure_fraction=need,erasure_other_memory_leak=leak,tested_mixtures=NQ,mean_mixture_size=float(np.mean(np.count_nonzero(Q,axis=0))),memory_count=M,optimizer_updates=0,code_sum_and_decode_macs=0,raw_vector_add_macs=NQ*float(np.mean(np.count_nonzero(Q,axis=0)))*D,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    Vmean=np.repeat(V.mean(1,keepdims=True),M,axis=1);mix,single,spec,cancel,need,leak=evalbank(Vmean,V,F,Qall)
    meanp=pack(shared_mean_trace=V.mean(1).astype(np.float32),memory_ids=np.arange(M,dtype=np.int8),fisher_diag=F.astype(np.float32))
    rows.append(dict(seed=SEED,method='shared_mean_trace',private_rank=1,serialized_bytes=len(meanp),payload_sha256=hashlib.sha256(meanp).hexdigest(),mixture_composition_nrmse=mix,single_memory_reactivation_nrmse=single,specificity_top1=spec,signed_cancellation_error=cancel,necessity_erasure_fraction=need,erasure_other_memory_leak=leak,tested_mixtures=NQ,mean_mixture_size=float(np.mean(np.count_nonzero(Q,axis=0))),memory_count=M,optimizer_updates=0,code_sum_and_decode_macs=NQ*D,raw_vector_add_macs=0,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run():print(row)
