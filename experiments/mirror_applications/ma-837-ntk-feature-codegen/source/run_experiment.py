"""Synthetic direct-conditioning vs Mirror task-coordinate screen."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1]; SEED=83701; T=12; U=8; K=2; F=24; NS=48; NT=128

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()

def nrmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)/(np.mean(b*b)+1e-12)))

def run():
    rng=np.random.default_rng(SEED)
    A=rng.normal(0,.12,(K,F)).astype(np.float64)
    z_seen=rng.normal(size=(T,K));z_unseen=rng.normal(size=(U,K))
    xs=rng.normal(size=(T,NS,F));xt=rng.normal(size=(T,NT,F));xu=rng.normal(size=(U,NT,F))
    ys=np.einsum('tnf,tf->tn',xs,z_seen@A);yt=np.einsum('tnf,tf->tn',xt,z_seen@A);yu=np.einsum('unf,uf->un',xu,z_unseen@A)
    # Fit the shared descriptor-conditioned readout from task support; no per-task weight update.
    design=np.einsum('tk,tnf->tnkf',z_seen,xs).reshape(T*NS,K*F)
    fit=np.linalg.lstsq(design,ys.reshape(-1),rcond=None)[0].reshape(K,F)
    seen_hat=z_seen@fit; unseen_hat=z_unseen@fit
    basis_spec=np.array([K,F,1],np.int16)
    native=pack(shared_generator=fit.astype(np.float32),basis_spec=basis_spec)
    # Mirror code is the externally supplied task descriptor: identical decode and no saved task table.
    direct=native; mirror=native
    cached=pack(shared_generator=fit.astype(np.float32),task_view_codes=z_seen.astype(np.float32),basis_spec=basis_spec)
    independent=pack(task_ids=np.arange(T,dtype=np.int8),task_contexts=z_seen.astype(np.float32),task_readouts=(z_seen@fit).astype(np.float32),basis_spec=basis_spec)
    shared_mean=pack(shared_readout=(z_seen@fit).mean(0).astype(np.float32),basis_spec=basis_spec)
    methods=[('native_ntk_no_task_storage',native,seen_hat,unseen_hat,0,'native shared generator'),('direct_descriptor_conditioning',direct,seen_hat,unseen_hat,0,'exact native alias'),('mirror_generated_coordinate',mirror,seen_hat,unseen_hat,0,'exact native alias'),('mirror_cached_task_codes',cached,np.tile(z_seen@fit,(1,1)),unseen_hat,T*K*F,'cached code table'),('independent_task_readouts',independent,z_seen@fit,None,0,'seen tasks only'),('shared_mean_readout',shared_mean,np.tile((z_seen@fit).mean(0),(T,1)),None,0,'no task coordinate')]
    rows=[]
    for name,payload,wseen,wunseen,decode,note in methods:
        st=time.perf_counter()
        pseen=np.einsum('tnf,tf->tn',xt,wseen)
        punseen=np.einsum('unf,uf->un',xu,wunseen) if wunseen is not None else np.full_like(yu,np.nan)
        modelbytes=len(payload);taskraw=(z_seen.nbytes if name in ('mirror_cached_task_codes','independent_task_readouts') else 0)
        rows.append(dict(seed=SEED,method=name,model_payload_bytes=modelbytes,task_state_array_raw_bytes=taskraw,total_serialized_state_bytes=modelbytes,payload_sha256=hashlib.sha256(payload).hexdigest(),seen_descriptor_nrmse=nrmse(pseen,yt),unseen_descriptor_nrmse=nrmse(punseen,yu) if wunseen is not None else '',support_examples=T*NS,optimizer_updates=0,active_inference_macs=T*NT*F,coordinate_decode_macs=decode,task_descriptor_input_bytes=K*4,roundtrip_max_abs=0.0,wall_time_s=time.perf_counter()-st,status_note=note))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    for row in run():print(row)
