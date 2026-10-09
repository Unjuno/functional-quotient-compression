#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads'
DEV=[47300,47301];FRESH=[47310,47311,47312];SEEDS=[0,1,2];D=16;L=4;E=64

def make_world(world,seed):
    bases_k=[];bases_v=[]
    for layer in range(L):
        g=torch.Generator().manual_seed(world*100003+seed*7919+layer*53+1)
        q,_=torch.linalg.qr(torch.randn(D,2,generator=g));bases_k.append(q)
        g=torch.Generator().manual_seed(world*31337+seed*199+layer*71+2)
        q,_=torch.linalg.qr(torch.randn(D,2,generator=g));bases_v.append(q)
    bk=torch.stack(bases_k);bv=torch.stack(bases_v)
    g=torch.Generator().manual_seed(world*99131+seed*71+17)
    ak=torch.rand(E,L,generator=g)*2*math.pi
    av=torch.rand(E,L,generator=torch.Generator().manual_seed(world*1217+seed*79+3))*2*math.pi
    ck=torch.stack([ak.cos(),ak.sin()],-1);cv=torch.stack([av.cos(),av.sin()],-1)
    keys=torch.einsum('elc,ldc->eld',ck,bk)
    scales=torch.tensor([1.0+0.1*l for l in range(L)])
    vals=torch.einsum('elc,ldc->eld',cv,bv)*scales[None,:,None]
    return bk,bv,ak,av,keys,vals,scales

def payload(method,bk,bv,ak,av,keys,vals,scales,n):
    if method=='no_edit': return b''
    o={'format':'ma473-v1','method':method,'n':n,'fact_ids':torch.arange(n,dtype=torch.int32)}
    if method=='memit_factors': o.update(keys=keys[:n].clone(),values=vals[:n].clone())
    elif method=='generic_coeff':
        o.update(key_basis=bk.clone(),value_basis=bv.clone(),key_coeff=torch.stack([ak[:n].cos(),ak[:n].sin()],-1),value_coeff=torch.stack([av[:n].cos(),av[:n].sin()],-1)*scales[None,:,None],scales=scales.clone())
    elif method=='mirror': o.update(key_basis=bk.clone(),value_basis=bv.clone(),key_angle=ak[:n].clone(),value_angle=av[:n].clone(),scales=scales.clone())
    b=io.BytesIO();torch.save(o,b);return b.getvalue()

def decode(method,p):
    if method=='no_edit': return None,None
    o=torch.load(io.BytesIO(p),map_location='cpu',weights_only=False)
    if method=='memit_factors': return o['keys'],o['values']
    if method=='generic_coeff':
        k=torch.einsum('elc,ldc->eld',o['key_coeff'],o['key_basis'])
        v=torch.einsum('elc,ldc->eld',o['value_coeff'],o['value_basis'])
        return k,v
    k=torch.stack([o['key_angle'].cos(),o['key_angle'].sin()],-1)
    v=torch.stack([o['value_angle'].cos(),o['value_angle'].sin()],-1)
    return torch.einsum('elc,ldc->eld',k,o['key_basis']),torch.einsum('elc,ldc->eld',v,o['value_basis'])*o['scales'][None,:,None]

def evaluate(keys,vals,kh,vh,n):
    if kh is None:return 1.0,1.0,0.0,0.0
    eff=[];loc=[];merged=[];g=torch.Generator().manual_seed(473991);q=torch.randn(32,D,generator=g)
    for i in range(n):
      for l in range(L):
        k=keys[i,l];v=vals[i,l];dk=kh[i,l];dv=vh[i,l]
        W=torch.outer(dv,dk)/(dk@dk)
        eff.append(float((W@k-v).norm()/(v.norm()+1e-12)))
        x=q-(q@k)[:,None]*k[None,:]/(k@k)
        loc.append(float((x@W.T).norm()/(x.norm()*v.norm()+1e-12)))
        Wall=torch.zeros(D,D)
        for j in range(n): Wall+=torch.outer(vh[j,l],kh[j,l])/(kh[j,l]@kh[j,l])
        merged.append(float(((Wall@k-v).norm()/(v.norm()+1e-12))))
    return statistics.mean(eff),max(eff),statistics.mean(loc),statistics.mean(merged)

def run(phase):
    worlds=DEV if phase=='development' else FRESH
    if phase=='development':
      (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'selection':'fixed four-layer rank-one orbit; no tuning'},indent=2)+'\n');print(json.dumps({'phase':phase,'worlds':len(worlds)*len(SEEDS)}));return
    rows=[]
    for w in worlds:
      for s in SEEDS:
        bk,bv,ak,av,keys,vals,scales=make_world(w,s)
        for method in ('no_edit','memit_factors','generic_coeff','mirror'):
          for n in (1,8,32,64):
            p=payload(method,bk,bv,ak,av,keys,vals,scales,n);kh,vh=decode(method,p);eff,effmax,loc,merge=evaluate(keys[:n],vals[:n],kh,vh,n)
            path=PAY/f'{w}_{s}_{method}_N{n}.pt'
            if p:path.write_bytes(p)
            times=[]
            for _ in range(30):
              t=time.perf_counter();payload(method,bk,bv,ak,av,keys,vals,scales,n);times.append(time.perf_counter()-t)
            # One selected edit through all four linear mediating layers.
            q=keys[0,0].clone();t=time.perf_counter()
            for _ in range(100):
              if kh is not None:
                for l in range(L): q=q+(torch.outer(vh[0,l],kh[0,l])/(kh[0,l]@kh[0,l]))@q
            query=(time.perf_counter()-t)/100
            rows.append({'world':w,'seed':s,'method':method,'n':n,'standalone_efficacy_nrmse_mean':eff,'standalone_efficacy_nrmse_max':effmax,'specificity_drift_mean':loc,'merged_bank_error_mean':merge,'payload_bytes':len(p),'bytes_per_edit':len(p)/n,'layerwise_decode_MAC_per_edit':L*D*4,'rankone_apply_MAC_per_layer':2*D*D,'payload_encode_seconds_per_edit':statistics.median(times)/n,'selected_query_seconds':query,'hash':hashlib.sha256(p).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])) if p else 'empty'})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run(a.parse_args().phase)
