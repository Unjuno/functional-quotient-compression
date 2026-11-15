# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Family-informed positive control: can 32 labelled pairs identify the true function?

This has a privileged formula-family prior and is NOT a matched-byte reference
for general shared representations. True coefficients are never used in fitting.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from a12 import FAMILIES, K_NEW, make_support, make_world


def predict_family(family:str,x:np.ndarray,params:tuple[float,...])->np.ndarray:
    xx=np.asarray(x,dtype=np.float64).reshape(-1)
    if family=='tent':
        c,w=params
        return np.maximum(0.,1.-np.abs(xx-c)/w)
    if family=='sine':
        amp,freq,phase=params
        return amp*np.sin(freq*xx+phase)
    raise ValueError('family')


def fit_family_from_support(family:str,x:np.ndarray,y:np.ndarray)->tuple[float,...]:
    xx=np.asarray(x,dtype=np.float64).reshape(-1)
    yy=np.asarray(y,dtype=np.float64).reshape(-1)
    if len(xx)!=len(yy) or len(xx)<8 or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError('support arrays')
    if family=='tent':
        cgrid=np.linspace(-.76,.76,77)
        wgrid=np.linspace(.16,.52,37)
        # Structured coarse grid followed by bounded least squares, no teacher center/width.
        best=[]
        for c in cgrid:
            pred=np.maximum(0.,1.-np.abs(xx[:,None]-c)/wgrid[None,:])
            scores=((pred-yy[:,None])**2).mean(axis=0)
            j=int(np.argmin(scores));best.append((float(scores[j]),float(c),float(wgrid[j])))
        starts=sorted(best)[:4]
        candidates=[]
        for _,c,w in starts:
            fit=least_squares(lambda z:predict_family('tent',xx,(z[0],z[1]))-yy,
                              x0=(c,w),bounds=([-.8,.1],[.8,.7]),
                              max_nfev=350,xtol=1e-13,ftol=1e-13,gtol=1e-13)
            est=tuple(float(x) for x in fit.x)
            candidates.append((np.mean((predict_family('tent',xx,est)-yy)**2),est))
        return min(candidates,key=lambda z:z[0])[1]
    if family=='sine':
        def lin_fit(freq):
            t=np.column_stack((np.sin(freq*xx),np.cos(freq*xx)))
            coef=np.linalg.lstsq(t,yy,rcond=None)[0]
            mse=np.mean((t@coef-yy)**2)
            return float(mse),coef
        grid=np.linspace(.35,2.85,301)
        vals=[lin_fit(f)[0] for f in grid]
        idx=int(np.argmin(vals))
        lo=grid[max(0,idx-2)];hi=grid[min(len(grid)-1,idx+2)]
        opt=minimize_scalar(lambda f:lin_fit(float(f))[0],bounds=(lo,hi),method='bounded',
                            options={'xatol':1e-13,'maxiter':200})
        freq=float(opt.x);_,coef=lin_fit(freq)
        amp=float(np.hypot(coef[0],coef[1]))
        phase=float(np.arctan2(coef[1],coef[0]))
        return (amp,freq,phase)
    raise ValueError('unknown family')


def run_oracle(families:list[str],worlds:list[int],supports:list[int],out:Path)->list[dict]:
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    rows=[]
    for family in families:
        for world in worlds:
            w=make_world(world,family)
            for n in supports:
                xx,yy=make_support(world,family,n)
                preds={}
                for split in ('new_iid','new_ood'):
                    preds[split]=np.zeros_like(w[split+'_y'].numpy(),dtype=np.float64)
                train_mse=[]
                for t in range(K_NEW):
                    x=xx[t,:,0].numpy().astype(np.float64)
                    y=yy[t].numpy().astype(np.float64)
                    fit=fit_family_from_support(family,x,y)
                    train_mse.append(float(np.mean((predict_family(family,x,fit)-y)**2)))
                    for split in ('new_iid','new_ood'):
                        xnew=w[split+'_x'][:,0].numpy().astype(np.float64)
                        preds[split][:,t]=predict_family(family,xnew,fit)
                rr=dict(family=family,world=world,support_n=n,mean_support_mse=float(np.mean(train_mse)),
                        formula_prior='KNOWN: tent 2 scalars / sine 3 scalars')
                for split in ('new_iid','new_ood'):
                    pred=preds[split]
                    target=w[split+'_y'].numpy().astype(np.float64)
                    var=np.mean((target-target.mean(axis=0,keepdims=True))**2)
                    rr[split+'_nrmse']=float(np.sqrt(np.mean((pred-target)**2)/var))
                rows.append(rr)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=run_oracle(list(FAMILIES),[1200,1201,1202],[32,256],a.out)
    for family in FAMILIES:
        for n in (32,256):
            sub=[r for r in rows if r['family']==family and r['support_n']==n]
            print('ORACLE',family,'n',n,'mean IID',round(np.mean([r['new_iid_nrmse'] for r in sub]),6),
                  'mean OOD',round(np.mean([r['new_ood_nrmse'] for r in sub]),6),flush=True)
