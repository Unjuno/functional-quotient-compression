#!/usr/bin/env python3
"""Fit and evaluate frozen affine distribution-view controls from Moirai samples."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from run_experiment import ROOT,CACHE,DATA_REPO,DATA_REV,get_data

NVAR=14
METHODS=("zero_shot","global","frequency_only","variate_only","additive","mirror_factorized","seasonal_naive")

def load_split(name):
    out=[]
    for f in (0,1):
        z=np.load(ROOT/"runs"/f"{name}_frequency_{f}.npz")
        for i in range(len(z["origins"])):
            out.append({"f":f,"v":int(z["sensor_ids"][i]),"origin":int(z["origins"][i]),
                "samples":z["samples"][i].astype(np.float64),"target":z["targets"][i].astype(np.float64),
                "seed":int(z["batch_seeds"][i])})
    return out

def pack_train(rows):
    x=[];y=[];f=[];v=[]
    for r in rows:
        x.extend(r["samples"].mean(axis=0));y.extend(r["target"])
        f.extend([r["f"]]*len(r["target"]));v.extend([r["v"]]*len(r["target"]))
    return np.asarray(x),np.asarray(y),np.asarray(f),np.asarray(v)

def softplus(x):
    return np.logaddexp(0.0,x)

def decode(method,z,f,v):
    if method=="zero_shot": return np.ones_like(f,dtype=float),np.zeros_like(f,dtype=float)
    if method=="global": return np.full(len(f),np.exp(z[0])),np.full(len(f),z[1])
    if method=="frequency_only":
        return np.exp(z[f]),z[2+f]
    if method=="variate_only":
        return np.exp(z[v]),z[NVAR+v]
    if method=="additive":
        logscale=z[:15][0]+(z[1] if False else 0)  # overwritten below using reference-coded factors
        # z: 15 scale effects [base,freq1,variate1..13], then 15 offset effects.
        ls=np.full(len(f),z[0]); ls += (f==1)*z[1]
        ls += np.choose(v, [0.0]+[z[1+j] for j in range(1,14)])
        b=np.full(len(f),z[15]); b += (f==1)*z[16]
        b += np.choose(v, [0.0]+[z[16+j] for j in range(1,14)])
        return np.exp(ls),b
    if method=="mirror_factorized":
        u=z[:2]; w=z[2:2+NVAR]; a=z[2+NVAR:4+NVAR]; c=z[4+NVAR:]
        return softplus(u[f])*softplus(w[v]),a[f]+c[v]
    raise KeyError(method)

def fit_codes(rows):
    x,y,f,v=pack_train(rows)
    def solve(fun,z0): return least_squares(lambda z: fun(z)-y,z0,max_nfev=120,ftol=1e-8).x.tolist()
    codes={}
    codes["zero_shot"]=[]
    codes["global"]=solve(lambda z: np.exp(z[0])*x+z[1],[0.,0.])
    codes["frequency_only"]=solve(lambda z: np.exp(z[f])*x+z[2+f],[0.,0.,0.,0.])
    codes["variate_only"]=solve(lambda z: np.exp(z[v])*x+z[NVAR+v],[0.]* (2*NVAR))
    def additive_pred(z):
        s,b=decode("additive",z,f,v);return s*x+b
    codes["additive"]=solve(additive_pred,[0.]*30)
    def mirror_pred(z):
        s,b=decode("mirror_factorized",z,f,v);return s*x+b
    z0=[0.5413248546]* (2+NVAR)+[0.]*(2+NVAR)
    codes["mirror_factorized"]=solve(mirror_pred,z0)
    return codes

def seasonal_samples(rows):
    _,_,_,series=get_data()
    out=[]
    for r in rows:
        f=r["f"]; x=series[f]; h=len(r["target"]); period=h
        hist=x[r["origin"]-period:r["origin"],r["v"]]
        out.append(np.broadcast_to(hist[None,:],r["samples"].shape))
    return out

def score(rows,codes,split):
    naive=seasonal_samples(rows); per_method={}
    # MASE scale is calculated exclusively from the first 60% of each sensor series.
    _,_,_,series=get_data(); denom={}
    for f,x in series.items():
        for v in range(NVAR):
            train=x[:int(.60*len(x)),v]
            denom[f,v]=max(float(np.mean(np.abs(np.diff(train)))),1e-8)
    cell_rows=[]
    for method in METHODS:
      grouped={}
      for i,r in enumerate(rows):
        f,v=r["f"],r["v"]; raw=r["samples"]; y=r["target"]
        if method=="seasonal_naive": pred=naive[i]
        else:
            s,b=decode(method,np.asarray(codes.get(method,[]),dtype=float),np.full(raw.shape[0],f),np.full(raw.shape[0],v))
            pred=raw*s[:,None]+b[:,None]
        # Empirical CRPS from a finite ensemble; sorted-sample identity avoids O(S^2).
        ordered=np.sort(pred,axis=0); n=pred.shape[0]
        coef=(2*np.arange(1,n+1)-n-1).reshape(n,1)
        pair_term=(ordered*coef).sum(axis=0)/(n*n)
        crps=np.mean(np.abs(pred-y[None,:]),axis=0)-pair_term
        median=np.quantile(pred,.5,axis=0)
        mase=np.abs(median-y)/denom[f,v]
        quant=np.quantile(pred,[.1,.5,.9],axis=0)
        taus=np.asarray([.1,.5,.9])[:,None]
        err=y[None,:]-quant
        pin=np.maximum(taus*err,(taus-1)*err)
        acc=grouped.setdefault((f,v),{"crps":[],"mase":[],"pin":[],"abs_y":[],"n":0})
        acc["crps"].extend(crps.tolist());acc["mase"].extend(mase.tolist())
        acc["pin"].append(pin.sum());acc["abs_y"].append(np.abs(y).sum());acc["n"]+=len(y)
      for (f,v),a in grouped.items():
        cell_rows.append({"split":split,"method":method,"frequency":f,"variate":v,
            "crps":float(np.mean(a["crps"])),"mase":float(np.mean(a["mase"])),
            "wql":float(2*sum(a["pin"])/(3*max(sum(a["abs_y"]),1e-8))),"points":a["n"]})
    return cell_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--audit",action="store_true");args=ap.parse_args()
    if args.audit:
        codes=json.loads((ROOT/"runs/DEV_CODES.json").read_text())
        rows=load_split("audit"); result=score(rows,codes,"audit")
        (ROOT/"runs/AUDIT_METRICS.json").write_text(json.dumps(result,indent=2)+"\n")
        print("Scored frozen Moirai controls on saved audit samples.")
        return
    train=load_split("train"); val=load_split("validation")
    codes=fit_codes(train); metrics=score(val,codes,"validation")
    means={m:float(np.mean([x["crps"] for x in metrics if x["method"]==m])) for m in METHODS}
    selected=min(means,key=means.get)
    (ROOT/"runs/DEV_CODES.json").write_text(json.dumps(codes,indent=2)+"\n")
    (ROOT/"runs/DEV_METRICS.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (ROOT/"runs/DEV_SELECTION.json").write_text(json.dumps({"validation_macro_crps":means,"selected":selected},indent=2)+"\n")
    print(json.dumps({"validation_macro_crps":means,"selected":selected},indent=2))
if __name__=="__main__":main()
