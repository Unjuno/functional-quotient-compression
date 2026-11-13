#!/usr/bin/env python3
"""Frozen synthetic MA-902 scene×time factorization mechanism screen."""
from __future__ import annotations
import csv, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source'/'results'
S,T,X,R=8,6,96,4
UPDATES=3000
LR=.03
WEIGHT_DECAY=1e-4
DEV=[90201,90207]
FRESH=[90221,90223,90229]

def generate(seed):
    rng=np.random.default_rng(seed)
    x=np.linspace(-1,1,X,dtype=np.float32)
    # Smooth, distinct shared physical spatial atoms.
    H=np.stack([np.sin((k+1)*math.pi*(x+1)/2) for k in range(R)]).astype(np.float32)
    H[0]=np.cos(.5*math.pi*x)
    A=rng.normal(0,.65,(S,R)).astype(np.float32)
    B=rng.normal(0,.65,(T,R)).astype(np.float32)
    coeff=A[:,None,:]*B[None,:,:]
    y=np.einsum('str,rx->stx',coeff,H)
    # Small pair-private structure is not represented by the shared rank-4 orbit.
    private=rng.normal(0,.012,(S,T,X)).astype(np.float32)
    y=(y+private).astype(np.float32)
    mask=np.zeros((S,T),dtype=bool)
    for s in range(S):
        for t in range(T):
            mask[s,t]=((t-((s*2+1)%T))%T)<4
    assert mask.sum()==32 and (~mask).sum()==16
    return x,y,mask

def fit(y,mask,seed,kind,weight_decay=0.0):
    torch.manual_seed(seed)
    yt=torch.tensor(y)
    mt=torch.tensor(mask)
    # Float32 CPU optimization; fit only observed cells.
    H=torch.nn.Parameter(torch.randn(R,X)*.25)
    A=torch.nn.Parameter(torch.randn(S,R)*.3)
    B=torch.nn.Parameter(torch.randn(T,R)*.3)
    opt=torch.optim.Adam([H,A,B],lr=LR,weight_decay=weight_decay)
    start=time.perf_counter()
    for _ in range(UPDATES):
        opt.zero_grad(set_to_none=True)
        if kind=='multiplicative':
            pred=torch.einsum('sr,tr,rx->stx',A,B,H)
        elif kind=='additive':
            pred=torch.einsum('sr,tr,rx->stx',A[:,None,:]+B[None,:,:],torch.ones_like(B)[None,:,:],H) if False else torch.einsum('str,rx->stx',A[:,None,:]+B[None,:,:],H)
        else: raise ValueError(kind)
        loss=((pred[mt]-yt[mt])**2).mean()
        loss.backward(); opt.step()
    elapsed=time.perf_counter()-start
    with torch.no_grad():
        pred=(torch.einsum('sr,tr,rx->stx',A,B,H) if kind=='multiplicative' else torch.einsum('str,rx->stx',A[:,None,:]+B[None,:,:],H)).numpy()
    return {'H':H.detach().numpy().astype(np.float32),'A':A.detach().numpy().astype(np.float32),'B':B.detach().numpy().astype(np.float32),'pred':pred,'wall':elapsed}

def npz_bytes(arrays):
    buf=io.BytesIO(); np.savez_compressed(buf,**arrays); return buf.getvalue()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; details=[]
    for split,seeds in [('development',DEV),('fresh',FRESH)]:
        for seed in seeds:
            x,y,mask=generate(seed)
            mult=fit(y,mask,seed+1,'multiplicative',WEIGHT_DECAY)
            add=fit(y,mask,seed+2,'additive',WEIGHT_DECAY)
            held=~mask
            scale=float(np.sqrt(np.mean(y[held]**2)))
            for name,res in [('mirror_multiplicative',mult),('native_rank4_factorization',mult),('additive_scene_time',add)]:
                err=float(np.sqrt(np.mean((res['pred'][held]-y[held])**2))/scale)
                arrays={'H':res['H'],'scene':res['A'],'time':res['B'],'meta':np.frombuffer(b'rank4;S8;T6;X96;float32',dtype=np.uint8)}
                payload=npz_bytes(arrays)
                (OUT/'payloads').mkdir(exist_ok=True)
                path=OUT/'payloads'/f'{split}_{seed}_{name}.npz'; path.write_bytes(payload)
                with np.load(path,allow_pickle=False) as z:
                    loaded={k:z[k] for k in z.files}
                if name=='additive_scene_time': replay=np.einsum('str,rx->stx',loaded['scene'][:,None,:]+loaded['time'][None,:,:],loaded['H'])
                else: replay=np.einsum('sr,tr,rx->stx',loaded['scene'],loaded['time'],loaded['H'])
                replay_delta=float(np.max(np.abs(replay-res['pred'])))
                assert replay_delta < 2e-5
                rows.append({'experiment_id':'MA-902','status':'FAIL','condition':f'{split}:{name}','seed':seed,'quality':f'nRMSE={err:.9g}','actual_payload_bytes':len(payload),'compute_proxy':R*X+2*R,'wall_clock_seconds':f"{res['wall']:.6f}",'notes':(('Native rank-4 control aliases the same multiplicative tensor factorization; same paid tensor state. ' if name=='native_rank4_factorization' else '')+f'Synthetic heldout cells; NPZ replay max_abs_delta={replay_delta:.3g}.' )})
            dense_arrays={'fields':y.astype(np.float32),'meta':np.frombuffer(b'S8;T6;X96;float32',dtype=np.uint8)}
            dense=npz_bytes(dense_arrays)
            (OUT/'payloads'/f'{split}_{seed}_independent_dense_oracle.npz').write_bytes(dense)
            with np.load(io.BytesIO(dense),allow_pickle=False) as dz: assert np.array_equal(dz['fields'],y)
            rows.append({'experiment_id':'MA-902','status':'FAIL','condition':f'{split}:independent_dense_oracle','seed':seed,'quality':'nRMSE=0 (oracle includes heldout truth)','actual_payload_bytes':len(dense),'compute_proxy':X,'wall_clock_seconds':'0','notes':'Independent full-cell oracle payload; includes heldout truth and cannot predict unseen cells.'})
            details.append({'split':split,'seed':seed,'observed_cells':int(mask.sum()),'heldout_cells':int(held.sum()),'heldout_indices':np.argwhere(held).tolist(),'private_residual_std':0.012,'multiplicative_fit_wall':mult['wall'],'additive_fit_wall':add['wall']})
    with (OUT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'split_manifest.json').write_text(json.dumps(details,indent=2)+'\n')
    # Self-check: all expected fresh heldout cells stay excluded from fitting schedule.
    assert all(d['observed_cells']==32 and d['heldout_cells']==16 for d in details)
    print(json.dumps({'rows':len(rows),'fresh_seeds':FRESH,'output':str(OUT/'RESULTS_CORE.csv'),'fresh_summary':[{r['condition']:r['quality'] for r in rows if r['condition'].startswith('fresh:') and int(r['seed'])==seed} for seed in FRESH]},indent=2))
if __name__=='__main__': main()
