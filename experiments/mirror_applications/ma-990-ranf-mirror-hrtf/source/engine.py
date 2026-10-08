"""MA-990 CPU HRTF field screen; RANF neural training is not reproduced."""
from __future__ import annotations
import hashlib, json, struct, time, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import signal

# sofar 1.1.3 and spatialaudiometrics 0.0.8 reference NumPy aliases removed in v2.
if not hasattr(np, 'float_'): np.float_ = np.float64
import sofar as sf
from spatialaudiometrics import hrtf_metrics as metrics

DIRECTIONS=793; EARS=2; SAMPLES=256; RANK=8; SPARSE_K=4
SUPPORTS={3:(4,11,414),5:(0,4,8,203,612)}
TRAIN=tuple(range(1,31)); DEV=tuple(range(31,37)); FRESH=tuple(range(37,43))
METHODS=('nearest_measured_direction','retrieval_nearest_training_listener','shared_pca_rank8_dense_code','mirror_top4_sparse_code','ordinary_top4_sparse_pca_code','independent_full_listener_map')

def encode(meta,arrays):
    specs=[]; chunks=[]
    for name in sorted(arrays):
        src=np.asarray(arrays[name]); dtype='<u1' if name=='indices' else '<f4'
        a=np.ascontiguousarray(src,dtype=dtype);raw=a.tobytes()
        specs.append({'name':name,'shape':list(a.shape),'dtype':dtype,'nbytes':len(raw)});chunks.append(raw)
    header=json.dumps({'meta':meta,'arrays':specs},sort_keys=True,separators=(',',':')).encode()
    return b'MA990\x01'+struct.pack('<I',len(header))+header+b''.join(chunks)

def decode(blob):
    assert blob[:6]==b'MA990\x01'; n=struct.unpack('<I',blob[6:10])[0]
    h=json.loads(blob[10:10+n]);pos=10+n; arrays={}
    for s in h['arrays']:
        raw=blob[pos:pos+s['nbytes']];pos+=s['nbytes']
        arrays[s['name']]=np.frombuffer(raw,dtype=s['dtype']).copy().reshape(s['shape'])
    assert pos==len(blob)
    return h['meta'],arrays

def read_map(data_root,subject):
    p=Path(data_root)/f'P{subject:04}_FreeFieldCompMinPhase_48kHz.sofa'
    sofa=sf.read_sofa(str(p),verbose=False)
    ir=np.asarray(sofa.Data_IR,dtype=np.float32)
    loc=np.asarray(sofa.SourcePosition,dtype=np.float64)
    assert ir.shape==(DIRECTIONS,EARS,SAMPLES),(subject,ir.shape)
    assert loc.shape[0]==DIRECTIONS and float(sofa.Data_SamplingRate)==48000
    return ir,loc

def fit_shared(train_maps,rank=RANK):
    x=np.asarray(train_maps,dtype=np.float64).reshape(len(train_maps),-1)
    mean=x.mean(axis=0); centered=x-mean
    gram=centered@centered.T
    val,u=np.linalg.eigh(gram); order=np.argsort(val)[::-1][:rank]
    sig=np.sqrt(np.maximum(val[order],1e-20)); basis=(u[:,order].T@centered)/sig[:,None]
    return mean.reshape(DIRECTIONS,EARS,SAMPLES).astype(np.float32),basis.reshape(rank,DIRECTIONS,EARS,SAMPLES).astype(np.float32)

def cartesian(pos):
    az=np.deg2rad(pos[:,0]);el=np.deg2rad(pos[:,1])
    return np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)],axis=1)

def fit_dense(mean,basis,target,support,ridge):
    a=basis[:,support].reshape(basis.shape[0],-1).T.astype(np.float64)
    y=(target[support]-mean[support]).reshape(-1).astype(np.float64)
    return np.linalg.solve(a.T@a+ridge*np.eye(a.shape[1]),a.T@y)

def fit_omp(mean,basis,target,support,k=SPARSE_K):
    a=basis[:,support].reshape(basis.shape[0],-1).T.astype(np.float64)
    y=(target[support]-mean[support]).reshape(-1).astype(np.float64)
    residual=y.copy();chosen=[];coef=None
    for _ in range(k):
        score=np.abs(a.T@residual)
        if chosen: score[chosen]=-np.inf
        chosen.append(int(np.argmax(score)))
        coef=np.linalg.lstsq(a[:,chosen],y,rcond=None)[0]
        residual=y-a[:,chosen]@coef
    return np.asarray(chosen,dtype=np.uint8),np.asarray(coef,dtype=np.float32)

def predict_code(mean,basis,coef,indices=None):
    if indices is None: vec=mean.reshape(-1)+np.asarray(coef,dtype=np.float64)@basis.reshape(len(basis),-1)
    else: vec=mean.reshape(-1)+np.asarray(coef,dtype=np.float64)@basis[np.asarray(indices,dtype=int)].reshape(len(indices),-1)
    return vec.reshape(DIRECTIONS,EARS,SAMPLES).astype(np.float32)

def nearest_map(target,train_maps,support):
    y=target[support].astype(np.float64).reshape(-1)
    ds=np.asarray([np.mean((m[support].astype(np.float64).reshape(-1)-y)**2) for m in train_maps])
    idx=int(np.argmin(ds));return train_maps[idx],idx

def nearest_direction(target,loc,support):
    xyz=cartesian(loc);unseen=np.setdiff1d(np.arange(DIRECTIONS),support)
    pred=target.copy()
    dist=np.linalg.norm(xyz[unseen,None,:]-xyz[None,support,:],axis=2)
    which=np.argmin(dist,axis=1);pred[unseen]=target[np.asarray(support)[which]]
    return pred

def acoustic_metrics(target,pred,unseen):
    a=SimpleNamespace(hrir=np.asarray(target[unseen],dtype=np.float64),fs=48000)
    b=SimpleNamespace(hrir=np.asarray(pred[unseen],dtype=np.float64),fs=48000)
    # spatialaudiometrics 0.0.8 returns a Python list then divides by fs;
    # reproduce its documented MAXIACC estimator with a NumPy conversion.
    def itd_samples(hrir):
        wn=3000/(48000/2); b,a=signal.butter(10,wn); out=[]
        for loc in hrir:
            left=signal.lfilter(b,a,loc[0,:]);right=signal.lfilter(b,a,loc[1,:])
            corr=signal.correlate(np.abs(signal.hilbert(left)),np.abs(signal.hilbert(right)))
            out.append(int(np.argmax(np.abs(corr))-hrir.shape[2]))
        return np.asarray(out,dtype=np.float64)
    itd=float(np.mean(np.abs(itd_samples(a.hrir)-itd_samples(b.hrir)))/48000*1e6)
    ild=float(metrics.calculate_ild_difference(a,b))
    lsd=float(metrics.calculate_lsd_across_locations(a.hrir,b.hrir,a.fs)[0])
    return lsd,ild,itd

def shared_payload(mean,basis):
    return encode({'kind':'shared_hrtf_field','rank':len(basis),'directions':DIRECTIONS}, {'mean':mean,'basis':basis})

def personalization_payload(coef,indices=None,listener_id=0):
    # The same sparse coefficient state is serialized for Mirror and its ordinary null control.
    arrays={'coefficients':np.asarray(coef,dtype=np.float32)}
    if indices is not None: arrays['indices']=np.asarray(indices,dtype=np.uint8)
    return encode({'kind':'listener_code','codec':'sparse4' if indices is not None else 'dense8','listener_id':int(listener_id)},arrays)

def evaluate_one(subject,target,loc,train_maps,train_ids,mean,basis,sparsity,ridge):
    support=np.asarray(SUPPORTS[sparsity],dtype=int);unseen=np.setdiff1d(np.arange(DIRECTIONS),support)
    rows=[];timer=time.perf_counter();full=encode({'kind':'independent_full_listener','listener_id':subject,'directions':DIRECTIONS},{'map':target})
    retrieval,train_idx=nearest_map(target,train_maps,support)
    dense=fit_dense(mean,basis,target,support,ridge);indices,sparse=fit_omp(mean,basis,target,support)
    predictions={
      'nearest_measured_direction':nearest_direction(target,loc,support),
      'retrieval_nearest_training_listener':retrieval,
      'shared_pca_rank8_dense_code':predict_code(mean,basis,dense),
      'mirror_top4_sparse_code':predict_code(mean,basis,sparse,indices),
      'ordinary_top4_sparse_pca_code':predict_code(mean,basis,sparse,indices),
      'independent_full_listener_map':target,
    }
    fit_wall=time.perf_counter()-timer
    sp=shared_payload(mean,basis);db=encode({'kind':'retrieval_database','listeners':list(train_ids)},{'maps':np.asarray(train_maps,dtype=np.float32)})
    for method,pred in predictions.items():
        t=time.perf_counter();lsd,ild,itd=acoustic_metrics(target,pred,unseen);wall=time.perf_counter()-t
        if method in ('mirror_top4_sparse_code','ordinary_top4_sparse_pca_code'):
            code=personalization_payload(sparse,indices,subject)
        elif method=='shared_pca_rank8_dense_code':code=personalization_payload(dense,listener_id=subject)
        elif method=='independent_full_listener_map':code=full
        elif method=='retrieval_nearest_training_listener':code=encode({'kind':'retrieval_choice','listener_id':subject},{'index':np.asarray([train_idx],dtype=np.uint8)})
        else:code=encode({'kind':'calibration_support','listener_id':subject,'support_directions':support.tolist()},{'support_ir':target[support]})
        shared_bytes=len(sp) if method in ('shared_pca_rank8_dense_code','mirror_top4_sparse_code','ordinary_top4_sparse_pca_code') else 0
        db_bytes=len(db) if method=='retrieval_nearest_training_listener' else 0
        rows.append({'subject_id':f'P{subject:04}','sparsity':sparsity,'support_directions':len(support),'method':method,'retrieved_training_subject':f'P{train_ids[train_idx]:04}' if method=='retrieval_nearest_training_listener' else '',
          'shared_payload_bytes':shared_bytes,'personalized_payload_bytes':len(code),'retrieval_database_bytes':db_bytes,'independent_map_bytes':len(full),
          'lsd_db':lsd,'ild_mae_db':ild,'itd_mae_us':itd,'calibration_fit_wall_s':fit_wall,'metric_wall_s':wall,
          'active_compute_proxy':int(len(support)*EARS*SAMPLES*RANK + DIRECTIONS*EARS*SAMPLES*(RANK if method.startswith(('shared_pca','mirror','ordinary_top4')) else 1)),
          'payload_sha256':hashlib.sha256(code).hexdigest(),'shared_sha256':hashlib.sha256(sp).hexdigest() if shared_bytes else ''})
    return rows

def run(mode,data_root='/tmp/ma990_data',ridge=0.01):
    subjects=DEV if mode=='development' else FRESH
    train_maps=[];loc=None
    for s in TRAIN:
        m,l=read_map(data_root,s);train_maps.append(m);loc=l
    mean,basis=fit_shared(np.asarray(train_maps))
    out=[]
    for subject in subjects:
        target,loc=read_map(data_root,subject)
        for sparsity in (3,5):out.extend(evaluate_one(subject,target,loc,np.asarray(train_maps),TRAIN,mean,basis,sparsity,ridge))
    import csv
    dest=Path(__file__).resolve().parents[1]/'source'/('development.csv' if mode=='development' else 'fresh.csv')
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('development','fresh'));ap.add_argument('--data',default='/tmp/ma990_data');ap.add_argument('--ridge',type=float,default=.01);args=ap.parse_args()
    for x in run(args.mode,args.data,args.ridge): print(x)
