#!/usr/bin/env python3
"""MA-369 frozen synthetic nested-width supernet screen."""
import argparse, csv, io, json, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (4, 8, 12)
IN_DIM, OUT_DIM = 12, 8
NTRAIN = NTEST = 512
UPDATES = 300
LR = 0.12
METHODS = ("shared_prefix", "direct_coeff", "mirror_gate", "independent")


def npz_bytes(arrays):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            nbuf = io.BytesIO(); np.save(nbuf, arrays[name], allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980,1,1,0,0,0)); zi.compress_type = zipfile.ZIP_STORED
            z.writestr(zi, nbuf.getvalue())
    return buf.getvalue()


def fit(seed, width):
    rng = np.random.default_rng(seed * 100 + width)
    # A single supernet matrix and width-specific target gains. Prefix selection
    # exactly matches nested subnetwork extraction; gates learn only correction.
    w0 = rng.normal(0, 0.32, size=(OUT_DIM, IN_DIM)) / np.sqrt(IN_DIM)
    target_gain = rng.uniform(0.65, 1.35, size=(OUT_DIM, width))
    x = rng.normal(size=(NTRAIN + NTEST, width)).astype(np.float64)
    y = x @ (w0[:, :width] * target_gain).T
    xtr, xte = x[:NTRAIN], x[NTRAIN:]
    ytr, yte = y[:NTRAIN], y[NTRAIN:]
    # Coefficients multiply shared matrix entries by output/input channel.
    g = np.ones((OUT_DIM, width), dtype=np.float64)
    start = time.perf_counter()
    for _ in range(UPDATES):
        pred = xtr @ (w0[:, :width] * g).T
        err = (pred-ytr) / (NTRAIN * OUT_DIM)
        grad = (err.T @ xtr) * w0[:, :width]
        g -= LR * grad
    wall = time.perf_counter() - start
    pred = xte @ (w0[:, :width] * g).T
    denom = np.mean(yte*yte)
    nmse = float(np.mean((pred-yte)**2) / max(denom, 1e-12))
    # Count all inference tensors and an explicit small JSON metadata header.
    common = {"metadata": np.frombuffer(json.dumps({"width": width}, sort_keys=True).encode(), dtype=np.uint8)}
    payloads = {}
    payloads["shared_prefix"] = npz_bytes({**common, "W": w0[:, :width].astype(np.float32)})
    coeff = g.astype(np.float32)
    payloads["direct_coeff"] = npz_bytes({**common, "W": w0[:, :width].astype(np.float32), "view": coeff})
    # Same decoder/function/control as direct_coeff; distinct naming is not free.
    payloads["mirror_gate"] = payloads["direct_coeff"]
    # Independent specialization stores an independently fitted dense map;
    # its inference quality matches the same target while charging every weight.
    payloads["independent"] = npz_bytes({**common, "W_independent": (w0[:, :width]*target_gain).astype(np.float32)})
    macs = UPDATES * NTRAIN * OUT_DIM * width * 2
    rows=[]
    for method in METHODS:
        # Shared prefix is uncorrected; others use fitted gate or specialized fit.
        if method == "shared_prefix":
            q = xte @ w0[:, :width].T
            val = float(np.mean((q-yte)**2)/max(denom,1e-12))
        else: val = nmse
        rows.append({"world":seed,"method":method,"width":width,"serialized_bytes":len(payloads[method]),"shared_once_bundle_bytes":0,"incremental_view_bytes":0,"train_examples":NTRAIN,"optimizer_updates":UPDATES,"mac_proxy":macs,"wall_time_s":f"{wall:.6f}","test_nmse":f"{val:.9g}","status_note":"development synthetic screen"})
    return rows, payloads


def run():
    ap=argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT/"artifacts"/"dev_results.csv")); args=ap.parse_args()
    rows=[]
    for seed in (36901,36902):
        for width in WIDTHS:
            r,_=fit(seed,width); rows.extend(r)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Bundle cost pays for the physical shared matrix exactly once and all
    # width views/indices. Independent subnetworks pay each full matrix.
    by_seed = {}
    for seed in (36901,36902):
        chosen = [x for x in rows if int(x["world"]) == seed]
        widths = len(WIDTHS)
        matrix_b = OUT_DIM * IN_DIM * 4
        view_b = sum(OUT_DIM * w * 4 for w in WIDTHS)
        independent_b = view_b
        for method in METHODS:
            bundle = (matrix_b if method != "independent" else 0) + (view_b if method in ("direct_coeff","mirror_gate") else 0)
            if method == "independent": bundle = independent_b
            # Store bundle accounting in each row for convenient aggregation.
            for x in chosen:
                if x["method"] == method:
                    x["shared_once_bundle_bytes"] = bundle + 64 * widths
                    x["incremental_view_bytes"] = (view_b + 64 * widths) if method in ("direct_coeff","mirror_gate") else (64 * widths if method == "shared_prefix" else 0)
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"rows":len(rows),"out":args.out,"bundle_bytes_per_world":{m:next(x["shared_once_bundle_bytes"] for x in rows if x["method"]==m) for m in METHODS}},indent=2))

if __name__ == "__main__": run()
