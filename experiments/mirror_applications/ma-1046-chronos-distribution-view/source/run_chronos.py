#!/usr/bin/env python3
"""Frozen Chronos MA-1046 development and fresh-audit runner."""
import argparse
import csv
import json
import math
import os
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from chronos import ChronosPipeline
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MA1046_DATA", "/tmp/ma1046-data"))
MODEL = Path(os.environ.get(
    "MA1046_MODEL",
    "/home/agent/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots/29d808298f1a62493e7b9a5e08529d0d930fa189",
))
SAMPLES = 32
CTX = 512
BATCH = 8
AUDIT_SEEDS = [104604, 104605, 104606]
WEATHER_IDS = [182, 304, 363, 587, 731, 770, 1482, 1490, 1888, 2021, 2452, 2658]
DATASET_SPECS = {
    "ETTh1": {"freq_min": 60, "horizon": 24, "period": 24},
    "ETTm1": {"freq_min": 15, "horizon": 24, "period": 96},
    "Weather": {"freq_min": 1440, "horizon": 30, "period": 7},
    "AustralianElectricity": {"freq_min": 30, "horizon": 48, "period": 48},
}


def ett_series(name):
    path = DATA / f"{name}.csv"
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    arr = np.asarray([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float32)
    return [(names[j], arr[:, j]) for j in range(arr.shape[1])]


def tsf_selected(zip_path, member, indices):
    want = set(indices)
    out = {}
    seen = 0
    data_started = False
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        for line in raw:
            line = line.decode("utf-8").strip()
            if not data_started:
                if line == "@data":
                    data_started = True
                continue
            if not line:
                continue
            if seen in want:
                fields = line.split(":")
                values = np.fromstring(fields[-1], sep=",", dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite data in {member} series index {seen}")
                out[seen] = (fields[0], values)
            seen += 1
    if want - out.keys():
        raise ValueError(f"requested TSF indices unavailable in {member}: {sorted(want-out.keys())}")
    return seen, [(str(i) + ":" + out[i][0], out[i][1]) for i in indices]


def load_dataset(name):
    if name in ("ETTh1", "ETTm1"):
        return ett_series(name)
    if name == "Weather":
        count, items = tsf_selected(DATA / "weather_dataset.zip", "weather_dataset.tsf", WEATHER_IDS)
        if count != 3010:
            raise ValueError(f"Weather series count drift: expected 3010, got {count}")
        return items
    if name == "AustralianElectricity":
        count, items = tsf_selected(
            DATA / "australian_electricity_demand_dataset.zip",
            "australian_electricity_demand_dataset.tsf",
            list(range(5)),
        )
        if count != 5:
            raise ValueError(f"Australian Electricity series count drift: expected 5, got {count}")
        return items
    raise KeyError(name)


def spaced_origins(n, start, end, horizon, count):
    lo = max(CTX, int(start))
    hi = int(end) - horizon
    if hi < lo:
        return []
    if count == 1:
        return [int((lo + hi) // 2)]
    vals = np.linspace(lo, hi, count)
    origins = []
    for v in vals:
        x = int(round(float(v)))
        if not origins or x - origins[-1] >= horizon:
            origins.append(x)
    return origins


def make_windows(dataset, series, mode):
    spec = DATASET_SPECS[dataset]
    h = spec["horizon"]
    windows = []
    for sidx, (sid, values) in enumerate(series):
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite input: {dataset}/{sid}")
        n = len(values)
        if mode == "dev":
            fit_end = int(0.60 * n)
            val_end = int(0.80 * n)
            ranges = [
                ("fit", spaced_origins(n, CTX, fit_end, h, 4)),
                ("val", spaced_origins(n, fit_end, val_end, h, 2)),
            ]
        else:
            ranges = [("audit", spaced_origins(n, 0.80 * n, n, h, 3))]
        for split, origins in ranges:
            for origin in origins:
                context = values[max(0, origin - CTX):origin]
                target = values[origin:origin + h]
                if len(context) < min(CTX, h * 2) or len(target) != h:
                    continue
                windows.append({
                    "dataset": dataset,
                    "series": sid,
                    "series_index": sidx,
                    "origin": int(origin),
                    "split": split,
                    "frequency_minutes": spec["freq_min"],
                    "horizon": h,
                    "period": spec["period"],
                    "context": context,
                    "target": target,
                    "history": values[:int(0.80 * n) if mode == "audit" else int(0.60 * n)],
                })
    return windows


def infer_windows(pipeline, windows, seed):
    out = []
    decoder_batch_s = []
    for start in range(0, len(windows), BATCH):
        batch = windows[start:start + BATCH]
        contexts = [torch.as_tensor(w["context"], dtype=torch.float32) for w in batch]
        torch.manual_seed(seed + start * 13)
        t0 = time.perf_counter()
        pred = pipeline.predict(
            contexts,
            prediction_length=batch[0]["horizon"],
            num_samples=SAMPLES,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        ).numpy().astype(np.float32)
        decoder_batch_s.append(time.perf_counter() - t0)
        if pred.shape != (len(batch), SAMPLES, batch[0]["horizon"]):
            raise ValueError(f"unexpected forecast shape {pred.shape}")
        for w, samples in zip(batch, pred):
            out.append({**w, "samples": samples})
    return out, decoder_batch_s


def basis(samples):
    # samples: [N,K,H]. The normalized squared-residual basis is orthogonal to
    # the first standardized residual basis for each forecast lead.
    mean = samples.mean(axis=1, keepdims=True)
    z = samples - mean
    norm = np.sqrt(np.mean(z * z, axis=1, keepdims=True))
    safe = np.maximum(norm, 1e-8)
    u = z / safe
    h2 = u * u
    h2 = h2 - h2.mean(axis=1, keepdims=True)
    projection = np.mean(u * h2, axis=1, keepdims=True)
    h2 = h2 - projection * u
    hnorm = np.sqrt(np.mean(h2 * h2, axis=1, keepdims=True))
    v = h2 / np.maximum(hnorm, 1e-8)
    v = np.where(hnorm > 1e-7, v, 0.0)
    return mean, norm, u, v


def mirror_view(samples, theta):
    t0 = time.perf_counter()
    mean, norm, u, v = basis(samples)
    result = mean + norm * (math.cos(theta) * u + math.sin(theta) * v)
    return result.astype(np.float32), time.perf_counter() - t0


def hermite_control(samples, theta):
    # Additive normalized H2 residual; mathematically identical to the
    # Givens mixture for |theta|<pi/2 with alpha=tan(theta).
    mean, norm, u, v = basis(samples)
    alpha = math.tan(theta)
    result = mean + norm * (u + alpha * v) / math.sqrt(1.0 + alpha * alpha)
    return result.astype(np.float32)


def temperature_control(samples, log_temp):
    mean = samples.mean(axis=1, keepdims=True)
    return (mean + math.exp(log_temp) * (samples - mean)).astype(np.float32)


def affine_control(samples, shift, scale):
    return (shift + scale * samples).astype(np.float32)


def sample_crps(samples, y):
    first = np.mean(np.abs(samples - y[:, None, :]), axis=1)
    pair = np.mean(np.abs(samples[:, :, None, :] - samples[:, None, :, :]), axis=(1, 2))
    return np.mean(first - 0.5 * pair, axis=1)


def quantile_metrics(samples, y):
    taus = [0.1, 0.5, 0.9]
    losses = []
    for tau in taus:
        q = np.quantile(samples, tau, axis=1)
        e = y - q
        losses.append(2.0 * np.maximum(tau * e, (tau - 1.0) * e))
    wql_num = np.sum(np.stack(losses), axis=(0, 2))
    wql_den = 3.0 * np.sum(np.abs(y), axis=1)
    med = np.quantile(samples, 0.5, axis=1)
    low = np.quantile(samples, 0.05, axis=1)
    high = np.quantile(samples, 0.95, axis=1)
    coverage = np.mean((y >= low) & (y <= high), axis=1)
    return wql_num, wql_den, np.mean(np.abs(med - y), axis=1), coverage


def naive_forecast(w, series_values):
    h = len(w["target"])
    p = int(w["period"])
    origin = int(w["origin"])
    return np.asarray([series_values[origin - p + (j % p)] for j in range(h)], dtype=np.float32)


def metrics(records, method_samples):
    # method_samples is [N,K,H], aligned with records.
    report = {}
    per_record = []
    naive_crps = []
    for i, r in enumerate(records):
        s = method_samples[i:i+1]
        y = r["target"][None, :]
        c = float(sample_crps(s, y)[0])
        wql_num, wql_den, mae, cov = quantile_metrics(s, y)
        nf = naive_forecast(r, r["full_series"])
        nc = float(np.mean(np.abs(nf - r["target"])))
        scale = float(np.mean(np.abs(np.diff(r["history"], n=1)))) if False else float(
            np.mean(np.abs(r["history"][r["period"]:] - r["history"][:-r["period"]]))
        )
        per_record.append({"dataset": r["dataset"], "series": r["series"], "crps": c,
                           "wql_num": float(wql_num[0]), "wql_den": float(wql_den[0]), "mae": float(mae[0]),
                           "mase": float(mae[0] / max(scale, 1e-8)),
                           "coverage90": float(cov[0]), "naive_crps": nc})
    for dataset in sorted({r["dataset"] for r in records}):
        rows = [x for x in per_record if x["dataset"] == dataset]
        ncrps = float(np.mean([x["naive_crps"] for x in rows]))
        report[dataset] = {
            "windows": len(rows),
            "crps": float(np.mean([x["crps"] for x in rows])),
            "crps_vs_seasonal_naive": float(np.mean([x["crps"] for x in rows]) / max(ncrps, 1e-8)),
            "wql_0.1_0.5_0.9": float(sum(x["wql_num"] for x in rows) / max(sum(x["wql_den"] for x in rows), 1e-8)),
            "mase": float(np.mean([x["mase"] for x in rows])),
            "coverage90": float(np.mean([x["coverage90"] for x in rows])),
            "naive_mae": ncrps
        }
    return report, per_record


def theta_from_code(freq_min, theta_15, theta_60):
    x = math.log2(freq_min / 60.0)
    # Dev knots are x=-2 (15m), x=0 (60m); clamp outside.
    if x <= -2.0:
        return float(theta_15)
    if x >= 0.0:
        return float(theta_60)
    return float(theta_15 + (x + 2.0) / 2.0 * (theta_60 - theta_15))


def grouped(records, splits=None):
    ids = [i for i, r in enumerate(records) if splits is None or r["split"] in splits]
    return ids


def fit_codes(records, samples, indices):
    grid = np.round(np.arange(-0.60, 0.6001, 0.05), 8)
    freqs = sorted({records[i]["frequency_minutes"] for i in indices})
    theta = {}
    log_temp = {}
    affine = {}
    for freq in freqs:
        ids = [i for i in indices if records[i]["frequency_minutes"] == freq]
        ss = samples[ids]
        yy = np.stack([records[i]["target"] for i in ids])
        best = (float("inf"), 0.0)
        for a in grid:
            sv = mirror_view(ss, float(a))[0]
            score = float(np.mean(sample_crps(sv, yy)))
            if score < best[0]:
                best = (score, float(a))
        theta[freq] = best[1]
        best_t = (float("inf"), 0.0)
        for a in grid:
            sv = temperature_control(ss, float(a))
            score = float(np.mean(sample_crps(sv, yy)))
            if score < best_t[0]:
                best_t = (score, float(a))
        log_temp[freq] = best_t[1]
        med = np.median(ss, axis=1)
        mean = np.mean(ss, axis=(1, 2))
        target_mean = float(np.mean(yy))
        centered_pred = float(np.mean(ss))
        # Fit standard per-frequency least-squares location and scale on train targets.
        p = ss.reshape(-1)
        yflat = np.repeat(yy[:, None, :], SAMPLES, axis=1).reshape(-1)
        A = np.column_stack([p, np.ones_like(p)])
        scale, shift = np.linalg.lstsq(A, yflat, rcond=None)[0]
        affine[freq] = [float(shift), float(max(scale, 0.0))]
    return {"theta_by_frequency": {str(k): v for k,v in theta.items()},
            "log_temp_by_frequency": {str(k): v for k,v in log_temp.items()},
            "affine_by_frequency": {str(k): v for k,v in affine.items()},
            "grid": grid.tolist()}


def save_raw(path, records, samples, decoder_times, sample_seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = []
    for r in records:
        meta.append({k:r[k] for k in ("dataset","series","series_index","origin","split","frequency_minutes","horizon","period")})
    targets = np.stack([r["target"] for r in records]).astype(np.float32)
    histories = np.stack([r["history"][-max(1,int(r["period"])*2):] for r in records]).astype(np.float32)
    np.savez_compressed(path, samples=samples.astype(np.float32), targets=targets,
                        histories=histories, metadata=np.asarray(json.dumps(meta)),
                        decoder_batch_seconds=np.asarray(decoder_times,dtype=np.float64),
                        sample_seed=np.asarray(sample_seed,dtype=np.int64))


def run(mode):
    torch.set_num_threads(4)
    torch.manual_seed(104600 if mode == "dev" else AUDIT_SEEDS[0])
    load_t0=time.perf_counter()
    pipeline=ChronosPipeline.from_pretrained(MODEL,device_map="cpu",torch_dtype=torch.float32)
    model_load_s=time.perf_counter()-load_t0
    if mode == "dev":
        names=["ETTh1","ETTm1"]
        seeds=[104600]
    else:
        names=["Weather","AustralianElectricity"]
        seeds=AUDIT_SEEDS
    all_for_fit=[]
    summary={"mode":mode,"model_load_s":model_load_s,"model_parameters":sum(p.numel() for p in pipeline.model.parameters()),
             "model_revision":"29d808298f1a62493e7b9a5e08529d0d930fa189","chronos_forecasting":"1.5.3"}
    for name in names:
        series=load_dataset(name)
        base_windows=make_windows(name,series,mode)
        if not base_windows:
            raise ValueError(f"no valid windows for {name}/{mode}")
        # Restore full series only for seasonal-naive calculation; histories used for MASE never include target labels.
        for w in base_windows:
            w["full_series"]=series[w["series_index"]][1]
        for seed in seeds:
            records,times=infer_windows(pipeline,base_windows,seed)
            samples=np.stack([r["samples"] for r in records])
            if mode=="dev":
                outdir=ROOT/"runs"/"dev"
            else:
                outdir=ROOT/"runs"/"audit"
            save_raw(outdir/f"{name}_seed{seed}.npz",records,samples,times,seed)
            all_for_fit.append((name,seed,records,samples,times))
        summary.setdefault("window_counts",{})[name]=len(base_windows)
    (ROOT/"runs"/(mode.upper()+"_RUN_SUMMARY.json")).write_text(json.dumps(summary,indent=2)+"\n")
    if mode=="dev":
        dev=[]
        for name,seed,records,samples,times in all_for_fit:
            dev.extend(zip(records,samples))
        rec=[x[0] for x in dev]; ss=np.stack([x[1] for x in dev])
        train_ids=grouped(rec,{"fit"})
        val_ids=grouped(rec,{"val"})
        train_codes=fit_codes(rec,ss,train_ids)
        final_codes=fit_codes(rec,ss,train_ids+val_ids)
        (ROOT/"runs"/"DEV_CANDIDATE.json").write_text(json.dumps({"train_fit":train_codes,"final_80pct_fit":final_codes},indent=2)+"\n")
        dev_metrics={}
        for split,ids,codes in [("train",train_ids,train_codes),("validation",val_ids,train_codes)]:
            rr=[rec[i] for i in ids]; raw=ss[ids]
            theta={int(k):v for k,v in codes["theta_by_frequency"].items()}
            tv={int(k):v for k,v in codes["log_temp_by_frequency"].items()}
            av={int(k):v for k,v in codes["affine_by_frequency"].items()}
            m=np.stack([mirror_view(raw[j:j+1],theta[rr[j]["frequency_minutes"]])[0][0] for j in range(len(rr))])
            h=np.stack([hermite_control(raw[j:j+1],theta[rr[j]["frequency_minutes"]]) [0] for j in range(len(rr))])
            t=np.stack([temperature_control(raw[j:j+1],tv[rr[j]["frequency_minutes"]])[0] for j in range(len(rr))])
            a=np.stack([affine_control(raw[j:j+1],*av[rr[j]["frequency_minutes"]])[0] for j in range(len(rr))])
            dev_metrics[split]={}
            for method,preds in [("chronos",raw),("mirror",m),("hermite",h),("native_temperature",t),("native_affine",a)]:
                met,_=metrics(rr,preds);dev_metrics[split][method]=met
        (ROOT/"runs"/"DEV_METRICS.json").write_text(json.dumps(dev_metrics,indent=2)+"\n")
    return summary


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["dev","audit"],required=True)
    args=ap.parse_args()
    print(json.dumps(run(args.mode),indent=2))
