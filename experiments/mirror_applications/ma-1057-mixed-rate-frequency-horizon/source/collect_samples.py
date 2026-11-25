#!/usr/bin/env python3
"""Collect frozen official Moirai samples for preregistered PEMS-BAY cells."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

HERE=Path(__file__).resolve().parents[1]; OUT=HERE/'runs'; CACHE='/tmp/ma1057-hf'
MODEL='Salesforce/moirai-1.1-R-small'; MODEL_REV='0c24ab99db2c1a70ea2a0fc03bf113329772ac64'
DATA='jimmygao3218/PEMSBAY'; DATA_REV='c858397fb381d82c3d55ad0a81beb51a9bb9bdaa'
SENSORS=list(range(14,22)); FIT_CELLS=[(0,0),(0,1),(1,1)]; AUDIT_CELL=(1,0)
FREQ={0:{'name':'5-minute','context':288,'period':288},1:{'name':'15-minute','context':96,'period':96}}
HORIZON={0:{'name':'1-hour','steps':{0:12,1:4}},1:{'name':'24-hour','steps':{0:288,1:96}}}
NSAMPLES=32; torch.set_num_threads(1)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def causal_block_mean(arr):
    n=(len(arr)//3)*3
    return arr[:n].reshape(-1,3,arr.shape[-1]).mean(axis=1)

def get_data():
    path=hf_hub_download(DATA,'PEMS-BAY.h5',repo_type='dataset',revision=DATA_REV,cache_dir=CACHE)
    frame=pd.read_hdf(path,'speed'); arr=frame.to_numpy(dtype=np.float32)
    finite=np.isfinite(arr).all(axis=0); ids=[i for i in range(14,arr.shape[1]) if finite[i]][:8]
    if ids!=SENSORS: raise ValueError(f'predeclared sensor rule selected {ids}, expected {SENSORS}')
    five=arr[:,ids]
    # Exact causal bin mean, no interpolation: source samples at k,k+1,k+2 complete bin before output.
    fifteen=causal_block_mean(five)
    series={0:five,1:fifteen}
    for f,x in series.items():
        if not np.isfinite(x).all(): raise ValueError('non-finite selected series')
    return path,frame,series

def make_origins(n,ctx,h,split,f):
    n15=(52116//3); ntr15=int(n15*.60); nd15=int(n15*.80)
    ntr,nd=(ntr15*3,nd15*3) if f==0 else (ntr15,nd15)
    if split=='train': lo,hi,k=ctx,ntr-h,4
    elif split=='dev': lo,hi,k=ntr,nd-h,2
    else: lo,hi,k=nd,n-h,3
    if hi<lo: raise ValueError((n,ctx,h,split,lo,hi))
    vals=np.unique(np.rint(np.linspace(lo,hi,k)).astype(np.int64))
    assert len(vals)==k and np.all(vals>=lo) and np.all(vals+h<=({'train':ntr,'dev':nd,'audit':n}[split]))
    return vals.tolist()

def load_model(patch):
    module=MoiraiModule.from_pretrained(MODEL,revision=MODEL_REV,cache_dir=CACHE,local_files_only=True).eval()
    models={}
    for f in (0,1):
        for h in (0,1):
            models[(f,h)]=MoiraiForecast(prediction_length=HORIZON[h]['steps'][f],target_dim=1,
                feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0,context_length=FREQ[f]['context'],
                module=module,patch_size=patch,num_samples=NSAMPLES).eval()
    return module,models

def cells(split): return FIT_CELLS if split in ('train','dev') else [AUDIT_CELL]

def collect(split,seeds,patch):
    path,frame,series=get_data(); module,models=load_model(patch); OUT.mkdir(exist_ok=True)
    data_manifest={'repo':DATA,'revision':DATA_REV,'file':'PEMS-BAY.h5','sha256':sha(path),
        'bytes':Path(path).stat().st_size,'rows':len(frame),'columns':frame.shape[1],
        'selected_ids':SENSORS,'selected_columns':[str(frame.columns[i]) for i in SENSORS],
        'finite':True,'5m_rows':len(series[0]),'15m_rows':len(series[1]),
        'canonical_15m_split_boundaries_in_5m_rows':[int(3*int(len(series[1])*.6)),int(3*int(len(series[1])*.8))],
        'downsample':'aligned non-overlapping mean, coefficients [1/3,1/3,1/3], no padding; each completed aggregate uses only its three source values'}
    scales={}
    for f,x in series.items():
        n15=len(series[1]); ntr15=int(n15*.60); ntr=ntr15*3 if f==0 else ntr15
        period=FREQ[f]['period']; train=x[:ntr]
        scales[str(f)]={'mean':train.mean(0).tolist(),'std':(train.std(0)+1e-8).tolist(),
            'seasonal_mase_scale':np.mean(np.abs(train[period:]-train[:-period]),axis=0).tolist(),
            'train_end':ntr,'period':period}
    data_manifest['training_only_series_scales']=scales
    scales={}
    for f,x in series.items():
        n15=len(series[1]); ntr15=int(n15*.60); ntr=ntr15*3 if f==0 else ntr15
        period=FREQ[f]['period']; train=x[:ntr]
        scales[str(f)]={'mean':train.mean(0).tolist(),'std':(train.std(0)+1e-8).tolist(),
            'seasonal_mase_scale':np.mean(np.abs(train[period:]-train[:-period]),axis=0).tolist(),
            'train_end':ntr,'period':period}
    data_manifest['training_only_series_scales']=scales
    model_path=hf_hub_download(MODEL,'model.safetensors',revision=MODEL_REV,cache_dir=CACHE,local_files_only=True)
    config_path=hf_hub_download(MODEL,'config.json',revision=MODEL_REV,cache_dir=CACHE,local_files_only=True)
    model_manifest={'repo':MODEL,'revision':MODEL_REV,'weights_sha256':sha(model_path),
        'weights_bytes':Path(model_path).stat().st_size,'config_sha256':sha(config_path),
        'config_bytes':Path(config_path).stat().st_size,'parameter_count':sum(p.numel() for p in module.parameters()),
        'dtype':'float32','patch_size':patch,'samples':NSAMPLES,'uni2ts':'1.1.1','torch':torch.__version__}
    manifest={'experiment_id':'MA-1057','data':data_manifest,'model':model_manifest,'split':split,'seeds':seeds}
    if split=='train': (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    n5=len(series[0]); n15=len(series[1]); max_latency=[]
    for f,h in cells(split):
        x=series[f]; ctx=FREQ[f]['context']; steps=HORIZON[h]['steps'][f]
        origins=make_origins(len(x),ctx,steps,split,f); windows=[]
        for local_sensor,sensor in enumerate(SENSORS):
            for origin in origins:
                past=x[origin-ctx:origin,local_sensor]; target=x[origin:origin+steps,local_sensor]
                if len(past)!=ctx or len(target)!=steps: raise ValueError('window boundary')
                windows.append((local_sensor,sensor,origin,past,target))
        for seed in seeds:
            samples=[]; lat=[]; batch_seeds=[]
            model=models[(f,h)]
            for bi in range(0,len(windows),4):
                batch=windows[bi:bi+4]
                arr=torch.as_tensor(np.stack([w[3] for w in batch])[...,None],dtype=torch.float32)
                observed=torch.ones_like(arr,dtype=torch.bool); padding=torch.zeros(arr.shape[:2],dtype=torch.bool)
                batch_seed=int(seed+f*10000+h*1000+bi); torch.manual_seed(batch_seed)
                start=time.perf_counter()
                with torch.inference_mode(): pred=model(arr,observed,padding,num_samples=NSAMPLES)
                elapsed=time.perf_counter()-start; lat.append(elapsed/len(batch)); batch_seeds.append(batch_seed)
                q=pred[:,:,:,0] if pred.ndim==4 else pred
                samples.extend(q.cpu().numpy())
            max_latency.extend(lat)
            out={'samples':np.asarray(samples,dtype=np.float32),
                'targets':np.asarray([w[4] for w in windows],dtype=np.float32),
                'sensor_ids':np.asarray([w[1] for w in windows],dtype=np.int16),
                'origins':np.asarray([w[2] for w in windows],dtype=np.int32),
                'batch_seeds':np.repeat(np.asarray(batch_seeds,dtype=np.int64),4)[:len(windows)],
                'naive':np.asarray([x[w[2]-FREQ[f]['period']:w[2],w[0]] for w in windows],dtype=np.float32),
                'frequency_id':np.asarray(f,dtype=np.int8),'horizon_id':np.asarray(h,dtype=np.int8),
                'context':np.asarray(ctx,dtype=np.int16),'horizon_steps':np.asarray(steps,dtype=np.int16)}
            folder=OUT/split; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'frequency{f}_horizon{h}_patch{patch}_seed{seed}.npz',**out)
    summary={'split':split,'cell_count':len(cells(split)),'sensor_count':len(SENSORS),
        'forecast_windows_per_seed_cell':len(SENSORS)*({'train':4,'dev':2,'audit':3}[split]),
        'model_inference_p50_seconds_per_window':float(np.quantile(max_latency,.5)) if max_latency else 0.0,
        'model_inference_p95_seconds_per_window':float(np.quantile(max_latency,.95)) if max_latency else 0.0,
        'data':data_manifest,'model':model_manifest}
    (OUT/f'{split}_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('split',choices=['train','dev','audit']); ap.add_argument('--seeds',nargs='+',type=int); ap.add_argument('--patch-size',type=int,choices=[8,16,32],required=True)
    a=ap.parse_args(); seeds=a.seeds or ([105701] if a.split in ('train','dev') else [105702,105703]); collect(a.split,seeds,a.patch_size)
