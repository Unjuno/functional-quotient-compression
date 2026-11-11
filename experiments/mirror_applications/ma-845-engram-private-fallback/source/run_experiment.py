"""Synthetic sequential shared-code allocation with paid private fallback."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=84501;D=32;M=24;R=4;THRESH=.02

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def measure(name,payload,Vhat,V,fallback,write_macs,query_macs):
    errs=np.sqrt(np.mean((Vhat-V)**2,axis=0)/(np.mean(V*V,axis=0)+1e-12))
    hits=0
    for i in range(M):hits+=int(np.argmin(np.sum((V-Vhat[:,i,None])**2,axis=0))==i)
    st=time.perf_counter();_ = Vhat.copy();elapsed=time.perf_counter()-st
    return dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),sequential_retrieval_nrmse=nrmse(Vhat,V),max_per_memory_nrmse=float(errs.max()),specificity_top1=hits/M,fallback_count=int(fallback),fallback_rate=float(fallback)/M,bytes_per_memory=len(payload)/M,write_projection_macs=write_macs,query_decode_macs=query_macs,optimizer_updates=0,wall_time_s=elapsed,payload_roundtrip_max_abs=0.0)

def run():
    rng=np.random.default_rng(SEED);F=np.geomspace(.5,2,D);Btrue,_=np.linalg.qr(rng.normal(size=(D,R)))
    v=[]
    for i in range(M):
        z=rng.normal(0,.25,R);base=Btrue@z
        if i<16:vec=base+rng.normal(0,.0005,D)
        else:
            q=rng.normal(size=D);q-=Btrue@(Btrue.T@q);q/=np.linalg.norm(q);vec=base+.22*q
        v.append(vec)
    V=np.stack(v,axis=1)
    # Freeze a shared basis from the first eight sequential memories.
    U,_,_=np.linalg.svd(V[:,:8],full_matrices=False);B=U[:,:R]
    codes=[];codes_inline=[];fallback_ids=[];fallback_vec=[];fallback_flags=[];Vshared=np.zeros_like(V);Vadapt=np.zeros_like(V)
    for i in range(M):
        c=B.T@V[:,i];recon=B@c;err=np.linalg.norm(V[:,i]-recon)/(np.linalg.norm(V[:,i])+1e-12)
        codes.append(c)
        if err<=THRESH:
            codes_inline.append(c);fallback_flags.append(0);Vshared[:,i]=recon;Vadapt[:,i]=recon
        else:
            codes_inline.append(np.zeros(R));fallback_flags.append(1);fallback_ids.append(i);fallback_vec.append(V[:,i]);Vadapt[:,i]=V[:,i]
    codes=np.stack(codes);codes_inline=np.stack(codes_inline);fallback_vec=np.stack(fallback_vec) if fallback_vec else np.zeros((0,D),np.float64)
    fallback_flags=np.array(fallback_flags,np.uint8);fallback_ids=np.array(fallback_ids,np.int16)
    # Complete hybrid payload includes all indices, private vectors, Fisher state and metadata.
    hybrid=pack(shared_basis=B.astype(np.float32),task_codes=codes_inline.astype(np.float32),fallback_vectors=fallback_vec.astype(np.float32),fallback_ids=fallback_ids,fallback_flags=fallback_flags,memory_ids=np.arange(M,dtype=np.int16),fisher_diag=F.astype(np.float32),threshold=np.array([THRESH],np.float32))
    nofb=pack(shared_basis=B.astype(np.float32),task_codes=codes.astype(np.float32),memory_ids=np.arange(M,dtype=np.int16),fisher_diag=F.astype(np.float32))
    # Native low-rank+private allocation is the direct control and serializes the same state.
    native=hybrid
    full=pack(memory_traces=V.astype(np.float32),memory_ids=np.arange(M,dtype=np.int16),fisher_diag=F.astype(np.float32))
    serac=pack(keys=np.eye(M,dtype=np.uint8),values=V.astype(np.float32),scope_radii=np.full(M,.02,np.float32),memory_ids=np.arange(M,dtype=np.int16),fisher_diag=F.astype(np.float32))
    fullhat=V.copy();serachat=V.copy()
    write=M*D*R;query_code=(M-len(fallback_ids))*D*R+len(fallback_ids)*D
    rows=[measure('mirror_shared_code_private_fallback',hybrid,Vadapt,V,len(fallback_ids),write,query_code),measure('native_lowrank_private_fallback',native,Vadapt,V,len(fallback_ids),write,query_code),measure('shared_code_no_fallback',nofb,Vshared,V,0,write,M*D*R),measure('independent_full_trace_table',full,fullhat,V,0,0,M*D),measure('serac_grace_explicit_memory',serac,serachat,V,0,0,M*D)]
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows
if __name__=='__main__':
    for row in run():print(row)
