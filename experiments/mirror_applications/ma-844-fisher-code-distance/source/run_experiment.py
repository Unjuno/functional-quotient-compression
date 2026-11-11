"""Synthetic Euclidean-vs-Fisher update vector quantization."""
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];SEED=84401;D=32;K=6;NTR=24;NTE=48;ITER=50

def pack(**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);return b.getvalue()
def dist(A,B,metric=None):
    d=A[:,None,:]-B[None,:,:]
    return np.sum(d*d if metric is None else d*d*metric[None,None,:],axis=-1)
def kmeans(train,k,metric=None):
    # Deterministic farthest-point initialization under the candidate metric.
    centers=[train[0].copy()]
    for _ in range(1,k):centers.append(train[np.argmax(np.min(dist(train,np.stack(centers),metric),axis=1))].copy())
    C=np.stack(centers);labels=np.zeros(len(train),int)
    for _ in range(ITER):
        newlabels=np.argmin(dist(train,C,metric),axis=1)
        newC=C.copy()
        for j in range(k):
            if np.any(newlabels==j):newC[j]=train[newlabels==j].mean(0)
        if np.array_equal(labels,newlabels) and np.allclose(C,newC):break
        C,labels=newC,newlabels
    return C,labels
def nrmse(a,b,metric=None):
    e=(a-b)**2;w=1 if metric is None else metric
    return float(np.sqrt(np.mean(e*w)/(np.mean(b*b*w)+1e-12)))

def run():
    rng=np.random.default_rng(SEED);F=np.geomspace(.1,10,D);sf=np.sqrt(F);centers_z=rng.normal(0,.35,(K,D));true_centers=centers_z/sf
    labels=np.repeat(np.arange(K),NTE//K+NTR//K);assert len(labels)==NTE+NTR
    # Construct balanced 4-shot codebook data and 8 heldout tasks per cluster.
    train=[];test=[];ytest=[]
    for c in range(K):
        train.extend((centers_z[c]+rng.normal(0,.015,(NTR//K,D)))/sf)
        test.extend((centers_z[c]+rng.normal(0,.015,(NTE//K,D)))/sf)
        ytest.extend([c]*(NTE//K))
    train=np.asarray(train);test=np.asarray(test);ytest=np.asarray(ytest)
    rows=[]
    for name,metric in [('euclidean_kmeans',None),('mirror_fisher_kmeans',F),('native_fisher_kmeans',F)]:
        st=time.perf_counter();cent,train_labels=kmeans(train,K,metric);codes=np.argmin(dist(test,cent,metric),axis=1);pred=cent[codes];elapsed=time.perf_counter()-st
        fisher_err=nrmse(pred,test,F);euc_err=nrmse(pred,test);center_map=np.argmin(dist(cent,true_centers,metric),axis=1);acc=float(np.mean(center_map[codes]==ytest))
        payload=pack(codebook=cent.astype(np.float32),task_codes=codes.astype(np.uint8),fisher_diag=F.astype(np.float32),task_ids=np.arange(NTE,dtype=np.int16),metric_tag=np.array([0 if metric is None else 1],np.int8))
        if name=='native_fisher_kmeans':payload=mirror_payload
        else:mirror_payload=payload if name=='mirror_fisher_kmeans' else None
        rows.append(dict(seed=SEED,method=name,serialized_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),heldout_fisher_nrmse=fisher_err,heldout_euclidean_nrmse=euc_err,cluster_retrieval_accuracy=acc,heldout_tasks=NTE,codebook_fit_tasks=NTR,optimizer_updates=0,assignment_distance_macs=(NTR+NTE)*K*D*(2 if metric is not None else 1),inference_decode_macs=NTE*D,wall_time_s=elapsed,payload_roundtrip_max_abs=0.0,status_note='native Fisher metric control alias' if name=='native_fisher_kmeans' else ''))
    independent=pack(task_updates=test.astype(np.float32),fisher_diag=F.astype(np.float32),task_ids=np.arange(NTE,dtype=np.int16))
    rows.append(dict(seed=SEED,method='independent_full_updates',serialized_bytes=len(independent),payload_sha256=hashlib.sha256(independent).hexdigest(),heldout_fisher_nrmse=0.0,heldout_euclidean_nrmse=0.0,cluster_retrieval_accuracy=1.0,heldout_tasks=NTE,codebook_fit_tasks=NTR,optimizer_updates=0,assignment_distance_macs=0,inference_decode_macs=0,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    mean=np.repeat(train.mean(0)[None,:],NTE,axis=0);meanp=pack(shared_update=train.mean(0).astype(np.float32),fisher_diag=F.astype(np.float32),task_ids=np.arange(NTE,dtype=np.int16))
    rows.append(dict(seed=SEED,method='shared_mean_update',serialized_bytes=len(meanp),payload_sha256=hashlib.sha256(meanp).hexdigest(),heldout_fisher_nrmse=nrmse(mean,test,F),heldout_euclidean_nrmse=nrmse(mean,test),cluster_retrieval_accuracy=0.0,heldout_tasks=NTE,codebook_fit_tasks=NTR,optimizer_updates=0,assignment_distance_macs=0,inference_decode_macs=NTE*D,wall_time_s=0.0,payload_roundtrip_max_abs=0.0,status_note=''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows
if __name__=='__main__':
    for row in run():print(row)
