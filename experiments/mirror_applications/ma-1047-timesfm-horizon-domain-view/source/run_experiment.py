#!/usr/bin/env python3
"""Frozen TimesFM CPU forecast collection for MA-1047."""
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/tmp/ma1047-data")
CACHE = "/tmp/ma1047-hf"
REV = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTm1.csv",
}
CTX, H, CHANNELS = 512, 720, 7
HORIZONS = [96, 192, 336, 720]

def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", revision=REV, cache_dir=CACHE,
        local_files_only=True, torch_compile=False)
    model.compile(timesfm.ForecastConfig(
        max_context=CTX, max_horizon=H, normalize_inputs=True,
        per_core_batch_size=4, use_continuous_quantile_head=True,
        force_flip_invariance=False, infer_is_positive=False,
        fix_quantile_crossing=True))
    return model

def read_dataset(name):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / (name + ".csv")
    if not path.exists():
        urllib.request.urlretrieve(URLS[name], path)
    df = pd.read_csv(path)
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if arr.shape[1] != CHANNELS or not np.isfinite(arr).all():
        raise ValueError(f"Unexpected numeric columns in {name}: {arr.shape}")
    return arr

def make_origins(n, split):
    if split == "train":
        lo, hi, count = CTX, int(.60*n)-H, 4
    elif split == "validation":
        lo, hi, count = int(.60*n), int(.80*n)-H, 2
    else:
        lo, hi, count = int(.80*n), n-H, 3
    if hi < lo:
        raise ValueError((n, split, lo, hi))
    return np.unique(np.rint(np.linspace(lo, hi, count)).astype(int)).tolist()

def collect(model, split):
    out, latency = {}, []
    for domain, name in enumerate(("ETTh1", "ETTm1")):
        arr = read_dataset(name)
        for ch in range(CHANNELS):
            origins = make_origins(len(arr), split)
            records = [(o, arr[o-CTX:o,ch], arr[o:o+H,ch]) for o in origins]
            forecasts = []
            for i in range(0, len(records), 4):
                batch = records[i:i+4]
                t0 = time.perf_counter()
                point, _ = model.forecast(horizon=H, inputs=[r[1] for r in batch])
                elapsed = time.perf_counter()-t0
                latency.extend([elapsed/len(batch)]*len(batch))
                forecasts.extend(point)
            out[f"{domain}:{ch}"] = {
                "domain": domain, "channel": ch, "origins": origins,
                "prediction": np.asarray(forecasts,dtype=np.float32).tolist(),
                "target": np.asarray([r[2] for r in records],dtype=np.float32).tolist()
            }
    return {"split":split,"revision":REV,"context":CTX,"horizon":H,
            "windows":out,"model_p50_seconds":float(np.quantile(latency,.5)),
            "model_p95_seconds":float(np.quantile(latency,.95))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("split",choices=["train","validation","audit"])
    args=ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS","4")
    data=collect(load_model(),args.split)
    target=ROOT/"runs"/(args.split.upper()+"_FORECASTS.json")
    target.write_text(json.dumps(data,separators=(",",":")))
    print(json.dumps({"split":args.split,"windows_per_domain_channel":len(next(iter(data["windows"].values()))["origins"]),"model_p50_seconds":data["model_p50_seconds"],"model_p95_seconds":data["model_p95_seconds"]},indent=2))

if __name__ == "__main__":
    main()
