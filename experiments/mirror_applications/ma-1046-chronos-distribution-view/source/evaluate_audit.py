#!/usr/bin/env python3
"""Apply frozen MA-1046 calibration codes and summarize fresh Chronos forecasts."""
import json
import time
from pathlib import Path

import numpy as np

from run_chronos import (
    AUDIT_SEEDS, DATASET_SPECS, ROOT, affine_control, hermite_control,
    load_dataset, metrics, mirror_view, theta_from_code, temperature_control,
)

FREEZE = ROOT / "runs" / "DEV_FREEZE.json"
AUDIT = ROOT / "runs" / "audit"


def interp(code, freq_min):
    x = np.log2(freq_min / 60.0)
    x0, x1 = np.log2(15.0 / 60.0), 0.0
    v0, v1 = code["15"], code["60"]
    if x <= x0: return float(v0)
    if x >= x1: return float(v1)
    return float(v0 + (x - x0) / (x1 - x0) * (v1 - v0))


def load_records(dataset, seed):
    path = AUDIT / f"{dataset}_seed{seed}.npz"
    data = np.load(path, allow_pickle=False)
    series = load_dataset(dataset)
    metas = json.loads(str(data["metadata"]))
    rec = []
    for i, m in enumerate(metas):
        idx = int(m["series_index"])
        full = series[idx][1]
        history = full[:int(0.80 * len(full))]
        rec.append({
            **m, "target": data["targets"][i], "history": history,
            "full_series": full, "samples": data["samples"][i],
        })
    return rec, data["samples"], data["decoder_batch_seconds"], data["sample_seed"]


def transform_for(records, raw, codes, method):
    out=[]
    post_s=[]
    for i,r in enumerate(records):
        freq=int(r["frequency_minutes"])
        if method in ("mirror","hermite"):
            theta=theta_from_code(freq,codes["theta_15min"],codes["theta_1h"])
            t=time.perf_counter()
            pred=mirror_view(raw[i:i+1],theta)[0] if method=="mirror" else hermite_control(raw[i:i+1],theta)
            post_s.append(time.perf_counter()-t)
        elif method=="native_temperature":
            tau=interp(codes["logtemp"],freq)
            t=time.perf_counter()
            pred=temperature_control(raw[i:i+1],tau)
            post_s.append(time.perf_counter()-t)
        elif method=="native_affine":
            x=np.log2(freq/60.0)
            if x<=-2: shift,scale=codes["affine"]["15"]
            elif x>=0: shift,scale=codes["affine"]["60"]
            else:
                w=(x+2)/2
                s0,k0=codes["affine"]["15"];s1,k1=codes["affine"]["60"]
                shift=(1-w)*s0+w*s1;scale=(1-w)*k0+w*k1
            t=time.perf_counter()
            pred=affine_control(raw[i:i+1],shift,scale)
            post_s.append(time.perf_counter()-t)
        elif method=="chronos":
            pred=raw[i:i+1]
        elif method=="seasonal_naive":
            h=int(r["horizon"]);p=int(r["period"]);o=int(r["origin"])
            nf=np.asarray([r["full_series"][o-p+(j%p)] for j in range(h)],dtype=np.float32)
            pred=np.repeat(nf[None,None,:],raw.shape[1],axis=1)
            post_s.append(0.0)
        else:
            raise KeyError(method)
        out.append(pred[0])
    return np.stack(out),post_s


def series_normalized(records, per_record):
    out={}
    for ds in sorted({r["dataset"] for r in records}):
        ids=[i for i,r in enumerate(records) if r["dataset"]==ds]
        grouped={}
        for i in ids:
            key=str(records[i]["series"])
            grouped.setdefault(key,[]).append(i)
        out[ds]={}
        for key, ix in grouped.items():
            row=per_record[ix]
            out[ds][key]={
                k:float(np.mean([x[k] for x in row])) for k in
                ("crps","wql","mase","coverage90","naive_crps")
            }
    return out


def bootstrap_ratio(series_scores, method, baseline="chronos", reps=10000, seed=61046):
    rng=np.random.default_rng(seed)
    datasets=sorted(series_scores[method])
    vals=[]
    for _ in range(reps):
        ds_ratios=[]
        for ds in datasets:
            keys=sorted(series_scores[method][ds])
            chosen=rng.choice(keys,size=len(keys),replace=True)
            num=np.mean([series_scores[method][ds][k]["crps"] for k in chosen])
            den=np.mean([series_scores[baseline][ds][k]["crps"] for k in chosen])
            ds_ratios.append(num/max(den,1e-12))
        vals.append(float(np.mean(ds_ratios)))
    return [float(np.quantile(vals,0.025)),float(np.quantile(vals,0.975))]


def main():
    codes=json.loads(FREEZE.read_text())["selected_codes"]
    report={"protocol_status":"locked","audit_seeds":AUDIT_SEEDS,"metrics_by_seed":{},
            "decoder_latency_by_seed":{},"view_latency_by_seed":{}}
    all_records=[];all_methods={m:[] for m in ["chronos","mirror","hermite","native_temperature","native_affine","seasonal_naive"]}
    series_acc={m:{} for m in all_methods}
    for seed in AUDIT_SEEDS:
        report["metrics_by_seed"][str(seed)]={}
        report["decoder_latency_by_seed"][str(seed)]={}
        report["view_latency_by_seed"][str(seed)]={}
        for dataset in ("Weather","AustralianElectricity"):
            records,raw,batch_times,_=load_records(dataset,seed)
            all_records.extend(records)
            method_arrays={}
            for method in all_methods:
                preds,post=transform_for(records,raw,codes,method)
                method_arrays[method]=preds
                met,per=metrics(records,preds)
                report["metrics_by_seed"][str(seed)][dataset][method]=met
                all_methods[method].append((records,preds))
                # Keep each series' scores paired across methods, datasets and sampling seeds.
                grp=series_normalized(records,per)
                for ds, rows in grp.items():
                    series_acc[method].setdefault(ds,{})
                    for sid, vals in rows.items():
                        series_acc[method][ds].setdefault(sid,[]).append(vals)
                report["view_latency_by_seed"][str(seed)].setdefault(dataset,{})[method]={
                    "p50_ms":float(np.quantile(post,0.50)*1000) if post else 0.0,
                    "p95_ms":float(np.quantile(post,0.95)*1000) if post else 0.0
                }
            # Convert batch latency to per-forecast decode latency; batches are ordered and size <=8.
            ns=len(records); sizes=[min(8,ns-i) for i in range(0,ns,8)]
            per=[t/n for t,n in zip(batch_times,sizes)]
            report["decoder_latency_by_seed"][str(seed)][dataset]={
                "p50_ms":float(np.quantile(per,0.50)*1000),
                "p95_ms":float(np.quantile(per,0.95)*1000),
                "total_decode_s":float(np.sum(batch_times)),
                "forecast_windows":ns
            }
    # Average scores over the three independent sampling seeds at the series-block level.
    collapsed={m:{} for m in series_acc}
    for method,dsets in series_acc.items():
        for ds,rows in dsets.items():
            collapsed[method][ds]={}
            for sid,vals in rows.items():
                collapsed[method][ds][sid]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    report["aggregate"]={}
    for method,dsets in collapsed.items():
        report["aggregate"][method]={}
        for ds,rows in dsets.items():
            den=np.mean([v["naive_crps"] for v in rows.values()])
            report["aggregate"][method][ds]={
                "series":len(rows),
                "crps":float(np.mean([v["crps"] for v in rows.values()])),
                "crps_vs_seasonal_naive":float(np.mean([v["crps"] for v in rows.values()])/max(den,1e-12)),
                "wql_0.1_0.5_0.9":float(np.mean([v["wql"] for v in rows.values()])),
                "mase":float(np.mean([v["mase"] for v in rows.values()])),
                "coverage90":float(np.mean([v["coverage90"] for v in rows.values()]))
            }
    report["paired_series_bootstrap_10000"]={
        m:bootstrap_ratio(collapsed,m) for m in ["mirror","hermite","native_temperature","native_affine","seasonal_naive"]
    }
    report["mirror_vs_native_temperature_normalized_crps"]={}
    for ds in ("Weather","AustralianElectricity"):
        report["mirror_vs_native_temperature_normalized_crps"][ds]=(
            report["aggregate"]["mirror"][ds]["crps_vs_seasonal_naive"] /
            max(report["aggregate"]["native_temperature"][ds]["crps_vs_seasonal_naive"],1e-12)
        )
    out=AUDIT/"AUDIT_METRICS.json";out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["aggregate"],indent=2))


if __name__=="__main__":
    main()
