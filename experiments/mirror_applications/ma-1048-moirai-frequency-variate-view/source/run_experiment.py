#!/usr/bin/env python3
"""Collect frozen Moirai predictive samples for PEMS-BAY frequency/variate pairs."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

ROOT=Path(__file__).resolve().parents[1]
REPO="Salesforce/moirai-1.1-R-small"
REV="0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
DATA_REPO="jimmygao3218/PEMSBAY"
DATA_REV="c858397fb381d82c3d55ad0a81beb51a9bb9bdaa"
CACHE="/tmp/ma1048-hf"
CTX=512
N_SAMPLES=32
FIT_PAIRS={0:[0,2,4,6,8,10,12,13],1:[1,3,5,7,9,11,12,13]}
AUDIT_PAIRS={0:[1,3,5,7,9,11],1:[0,2,4,6,8,10]}

def get_data():
    p=hf_hub_download(DATA_REPO,"PEMS-BAY.h5",repo_type="dataset",revision=DATA_REV,cache_dir=CACHE)
    frame=pd.read_hdf(p,"speed")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("PEMS-BAY source contains missing or non-finite values")
    cols=list(frame.columns[:14])
    five=frame.iloc[:,:14].to_numpy(dtype=np.float32)
    fifteen=five[:len(five)//3*3].reshape(-1,3,14).mean(axis=1)
    return p,frame.index,cols,{0:five,1:fifteen}

def get_origins(n,h,split):
    if split=="train":
        lo,hi,k=CTX+h,int(.60*n)-h,4
    elif split=="validation":
        lo,hi,k=int(.60*n),int(.80*n)-h,2
    else:
        lo,hi,k=int(.80*n),n-h,3
    if hi<lo: raise ValueError((n,h,split,lo,hi))
    return np.unique(np.rint(np.linspace(lo,hi,k)).astype(int)).tolist()

def load_model():
    module=MoiraiModule.from_pretrained(REPO,revision=REV,cache_dir=CACHE,local_files_only=True).eval()
    forecasts={}
    for f,h in [(0,288),(1,96)]:
        forecasts[f]=MoiraiForecast(prediction_length=h,target_dim=1,
            feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0,
            context_length=CTX,module=module,patch_size="auto",num_samples=N_SAMPLES).eval()
    return module,forecasts

def collect(split):
    path,index,cols,series=get_data()
    module,models=load_model()
    output={}; latency=[]
    for f in [0,1]:
        x=series[f]; h=288 if f==0 else 96
        pairs=FIT_PAIRS[f] if split in ("train","validation") else AUDIT_PAIRS[f]
        windows=[]
        for sensor in pairs:
            for origin in get_origins(len(x),h,split):
                past=x[origin-(CTX+h):origin,sensor]
                target=x[origin:origin+h,sensor]
                if len(past)!=CTX+h or len(target)!=h: raise ValueError("window bounds")
                windows.append((sensor,origin,past,target))
        preds=[]; batch_seeds=[]
        for bi in range(0,len(windows),4):
            batch=windows[bi:bi+4]
            arr=torch.as_tensor(np.stack([r[2] for r in batch])[...,None],dtype=torch.float32)
            observed=torch.ones_like(arr,dtype=torch.bool)
            padding=torch.zeros(arr.shape[:2],dtype=torch.bool)
            seed=104801+(0 if split=="train" else 100000 if split=="validation" else 200000)+f*10000+bi
            torch.manual_seed(seed)
            start=time.perf_counter()
            with torch.inference_mode():
                sample=models[f](arr,observed,padding,num_samples=N_SAMPLES)
            elapsed=time.perf_counter()-start
            latency.extend([elapsed/len(batch)]*len(batch))
            pred=sample[:,:, :,0] if sample.ndim==4 else sample
            preds.extend(pred.cpu().numpy())
            batch_seeds.extend([seed]*len(batch))
        output[f"frequency_{f}"]={
            "frequency_minutes":5 if f==0 else 15,"horizon":h,
            "samples":np.asarray(preds,dtype=np.float32),
            "targets":np.asarray([r[3] for r in windows],dtype=np.float32),
            "sensor_ids":np.asarray([r[0] for r in windows],dtype=np.int16),
            "origins":np.asarray([r[1] for r in windows],dtype=np.int32),
            "batch_seeds":np.asarray(batch_seeds,dtype=np.int64)}
    outdir=ROOT/"runs";outdir.mkdir(exist_ok=True)
    for key,val in output.items():
        np.savez_compressed(outdir/f"{split}_{key}.npz",**val)
    if split=="train":
        model_path=hf_hub_download(REPO,"model.safetensors",revision=REV,cache_dir=CACHE,local_files_only=True)
        config_path=hf_hub_download(REPO,"config.json",revision=REV,cache_dir=CACHE,local_files_only=True)
        manifest={"model":{"repo":REPO,"revision":REV,"weights_bytes":Path(model_path).stat().st_size,
            "weights_sha256":hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "config_bytes":Path(config_path).stat().st_size,
            "config_sha256":hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "parameter_count":sum(p.numel() for p in module.parameters()),"dtype":"float32"},
            "data":{"repo":DATA_REPO,"revision":DATA_REV,"file":"PEMS-BAY.h5",
            "bytes":Path(path).stat().st_size,"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "source_rows":len(index),"source_sensor_columns":len(pd.read_hdf(path,"speed").columns),
            "selected_sensor_ids":[int(c) for c in cols],"five_minute_rows":len(series[0]),
            "fifteen_minute_rows":len(series[1]),"finite_values":True},
            "software":{"uni2ts":"1.1.1","torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__},
            "sample_count":N_SAMPLES,"context":CTX,"raw_inputs_not_committed":True}
        (outdir/"INPUT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summary={"split":split,"forecast_p50_seconds_per_series":float(np.quantile(latency,.5)),
             "forecast_p95_seconds_per_series":float(np.quantile(latency,.95)),
             "windows_by_frequency":{str(f):len(output[f"frequency_{f}"]["origins"]) for f in [0,1]}}
    (outdir/f"{split.upper()}_RUN_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("split",choices=["train","validation","audit"])
    collect(ap.parse_args().split)
