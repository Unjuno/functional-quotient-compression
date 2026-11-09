#!/usr/bin/env python3
"""Cross-backbone shared bridge screen on synthetic linear controls."""
from __future__ import annotations
import csv,hashlib,io,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SEED=75501; S=3; B=4; IN=8; OUT=8; K=2
HELDOUT={(0,0),(1,2),(2,3)}
torch.set_num_threads(1)

def pack(**a):
    f=io.BytesIO();np.savez_compressed(f,**a);return f.getvalue()

def run():
    rng=np.random.default_rng(SEED)
    enc=rng.normal(scale=.45,size=(S,K,IN)).astype(np.float32)
    dec=rng.normal(scale=.45,size=(B,OUT,K)).astype(np.float32)
    target=np.einsum('bok,ski->sboi',dec,enc)
    train=[(s,b) for s in range(S) for b in range(B) if (s,b) not in HELDOUT]
    test=sorted(HELDOUT)
    torch.manual_seed(SEED)
    et=torch.nn.Parameter(torch.randn(S,K,IN)*.08)
    dt=torch.nn.Parameter(torch.randn(B,OUT,K)*.08)
    opt=torch.optim.Adam([et,dt],lr=.025)
    tt=torch.tensor(target)
    start=time.perf_counter();updates=0
    for step in range(2500):
      opt.zero_grad(); loss=sum(torch.mean((dt[b]@et[s]-tt[s,b])**2) for s,b in train)/len(train)
      loss.backward();opt.step();updates+=1
    fit_ms=(time.perf_counter()-start)*1000
    learned=(dt.detach().numpy()[:,None,:,:] @ et.detach().numpy()[None,:,:,:]).transpose(1,0,2,3)
    # Simple controls are estimated only from observed training pair matrices.
    source_only=np.stack([np.mean([target[s,b] for s0,b in train if s0==s],axis=0) for s in range(S)])
    target_only=np.stack([np.mean([target[s,b] for s,b0 in train if b0==b],axis=0) for b in range(B)])
    ids=np.arange(S*B,dtype=np.int8).reshape(S,B)
    full=pack(pair_bridges=target,ids=ids)
    source_blob=pack(source_bridges=source_only,ids=np.arange(S,dtype=np.int8))
    target_blob=pack(backbone_bridges=target_only,ids=np.arange(B,dtype=np.int8))
    tower=pack(source_encoders=et.detach().numpy(),backbone_decoders=dt.detach().numpy(),source_ids=np.arange(S,dtype=np.int8),backbone_ids=np.arange(B,dtype=np.int8))
    rows=[]
    for name,blob,pred in [
      ('independent_pair_bridges',full,lambda s,b:target[s,b]),
      ('source_only_bridge',source_blob,lambda s,b:source_only[s]),
      ('backbone_only_bridge',target_blob,lambda s,b:target_only[b]),
      ('native_two_tower_bridge',tower,lambda s,b:learned[s,b]),
      ('mirror_source_x_backbone_views',tower,lambda s,b:learned[s,b])]:
      errs=[];t0=time.perf_counter()
      for s,b in test:errs.append(np.linalg.norm(pred(s,b)-target[s,b])/np.linalg.norm(target[s,b]))
      eval_ms=(time.perf_counter()-t0)*1000
      # Three heldout matrix queries, each either one dense bridge or two tower multiplies.
      macs=len(test)*IN*OUT if name not in ('native_two_tower_bridge','mirror_source_x_backbone_views') else len(test)*K*(IN+OUT)
      rows.append(dict(seed=SEED,method=name,heldout_pairs=';'.join(f'{s}:{b}' for s,b in test),serialized_bytes=len(blob),
          payload_sha256=hashlib.sha256(blob).hexdigest(),heldout_bridge_nrmse=float(np.mean(errs)),
          active_macs=macs,optimizer_updates=updates if name in ('native_two_tower_bridge','mirror_source_x_backbone_views') else 0,
          fit_wall_ms=fit_ms if name in ('native_two_tower_bridge','mirror_source_x_backbone_views') else 0,
          eval_wall_ms=eval_ms,status_note='exact native two-tower alias' if name=='mirror_source_x_backbone_views' else ''))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('fit updates',updates,'heldout',test)
if __name__=='__main__':run()
