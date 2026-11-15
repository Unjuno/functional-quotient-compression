#!/usr/bin/env python3
"""Fit preregistered affine head controls and select by development validation."""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from run_experiment import ROOT, DATA, HORIZONS, read_dataset

TRAIN=json.loads((ROOT/"runs/TRAIN_FORECASTS.json").read_text())
VAL=json.loads((ROOT/"runs/VALIDATION_FORECASTS.json").read_text())
DSETS=("ETTh1","ETTm1")

def cells(bundle):
    X=[];Y=[]
    for d in range(2):
      for hi,h in enumerate(HORIZONS):
        x=[];y=[]
        for ch in range(7):
          z=bundle["windows"][f"{d}:{ch}"]
          x.append(np.asarray(z["prediction"],dtype=np.float64)[:,:h].reshape(-1))
          y.append(np.asarray(z["target"],dtype=np.float64)[:,:h].reshape(-1))
        X.append(np.concatenate(x));Y.append(np.concatenate(y))
    return X,Y

def fit(X,Y):
    # independent cell affine heads
    cell=np.stack([np.linalg.lstsq(np.stack([x,np.ones_like(x)],1),y,rcond=None)[0] for x,y in zip(X,Y)]).reshape(2,4,2)
    xx=np.concatenate(X); yy=np.concatenate(Y)
    shared=np.linalg.lstsq(np.stack([xx,np.ones_like(xx)],1),yy,rcond=None)[0]
    # Native additive domain+horizon affine (5 scale and 5 bias coefficients).
    rows=[]; targets=[]
    for j,(x,y) in enumerate(zip(X,Y)):
      d,h=divmod(j,4)
      z=np.zeros((len(x),12)); z[:,0]=x
      if d: z[:,1]=x
      z[:,2+h]=x
      z[:,6]=1
      if d: z[:,7]=1
      z[:,8+h]=1
      rows.append(z); targets.append(y)
    beta=np.linalg.lstsq(np.concatenate(rows),np.concatenate(targets),rcond=None)[0]
    additive=np.zeros((2,4,2))
    for d in range(2):
      for h in range(4):
        additive[d,h]=[beta[0]+(beta[1] if d else 0)+beta[2+h],beta[6]+(beta[7] if d else 0)+beta[8+h]]
    # Factorized scale u[d]*v[h], additive bias a[d]+b[h].
    def coeff(z):
      u=z[:2];v=z[2:6];a=z[6:8];b=z[8:12]
      return u[:,None]*v[None,:],a[:,None]+b[None,:]
    def residual(z):
      s,b=coeff(z); return np.concatenate([s.flat[j]*x+b.flat[j]-y for j,(x,y) in enumerate(zip(X,Y))])
    z0=np.array([1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.])
    opt=least_squares(residual,z0,max_nfev=100,ftol=1e-8)
    s,b=coeff(opt.x); mirror=np.stack([s,b],axis=-1)
    baseline=np.zeros((2,4,2)); baseline[:,:,0]=1
    return {"timesfm":baseline.tolist(),"shared":np.tile(shared,(2,4,1)).tolist(),
            "additive":additive.tolist(),"independent":cell.tolist(),
            "mirror_factorized":mirror.tolist(),"factorized_parameters":opt.x.tolist(),
            "additive_parameters":beta.tolist()}

def score(bundle,heads):
    rows=[]
    for d,name in enumerate(DSETS):
      vals=read_dataset(name); period=24 if name=="ETTh1" else 96
      for ch in range(7):
        z=bundle["windows"][f"{d}:{ch}"]
        pred=np.asarray(z["prediction"],dtype=np.float64)
        tar=np.asarray(z["target"],dtype=np.float64)
        origins=z["origins"]
        scale=np.mean(np.abs(np.diff(vals[:int(.60*len(vals)),ch])))
        for hi,h in enumerate(HORIZONS):
          y=tar[:,:h]
          for method,head in heads.items():
            if np.asarray(head).shape != (2,4,2):
              continue
            a,b=np.asarray(head)[d,hi]
            yp=a*pred[:,:h]+b
            mase=np.mean(np.abs(yp-y))/max(scale,1e-8)
            smape=np.mean(2*np.abs(yp-y)/(np.abs(yp)+np.abs(y)+1e-6))*100
            rows.append({"dataset":name,"channel":ch,"horizon":h,"method":method,"mase":float(mase),"smape":float(smape)})
          naive=[]
          for origin in origins:
            hist=vals[origin-period:origin,ch]
            naive.append(np.resize(hist,h))
          yp=np.asarray(naive,dtype=np.float64)
          mase=np.mean(np.abs(yp-y))/max(scale,1e-8)
          smape=np.mean(2*np.abs(yp-y)/(np.abs(yp)+np.abs(y)+1e-6))*100
          rows.append({"dataset":name,"channel":ch,"horizon":h,"method":"seasonal_naive","mase":float(mase),"smape":float(smape)})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--audit",action="store_true"); args=ap.parse_args()
    if args.audit:
      heads=json.loads((ROOT/"runs/DEV_HEADS.json").read_text())
      bundle=json.loads((ROOT/"runs/AUDIT_FORECASTS.json").read_text())
      rows=score(bundle,heads)
      (ROOT/"runs/AUDIT_METRICS.json").write_text(json.dumps(rows,indent=2)+"\n")
      print("Saved frozen-code fresh audit metrics")
      return
    heads=fit(*cells(TRAIN)); rows=score(VAL,heads)
    methods=[m for m,h in heads.items() if np.asarray(h).shape==(2,4,2)]
    means={m:float(np.mean([r["mase"] for r in rows if r["method"]==m])) for m in methods}
    winner=min(means,key=means.get)
    (ROOT/"runs/DEV_HEADS.json").write_text(json.dumps(heads,indent=2)+"\n")
    (ROOT/"runs/DEV_SELECTION.json").write_text(json.dumps({"validation_mase":means,"selected":winner,"codes":heads[winner],"selection":"minimum macro mean MASE over frozen method list"},indent=2)+"\n")
    (ROOT/"runs/DEV_METRICS.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps({"validation_mase":means,"selected":winner},indent=2))
if __name__=="__main__": main()
