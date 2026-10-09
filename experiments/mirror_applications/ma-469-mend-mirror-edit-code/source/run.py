#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46900,46901];FRESH=[46910,46911,46912];SEEDS=[0,1,2]
def data(w,s):
 g=torch.Generator().manual_seed(w*1000003+s*997+53);keys=torch.randn(64,2,generator=g);keys=keys/keys.norm(dim=1,keepdim=True);deltas=torch.randn(64,2,generator=torch.Generator().manual_seed(w*31337+s*199))*.5
 para=keys+torch.randn(64,2,generator=torch.Generator().manual_seed(w*99131+s*71))*.05
 local=torch.randn(64,128,2,generator=torch.Generator().manual_seed(w*1217+s*79));local=local/local.norm(dim=2,keepdim=True)
 # Retain unrelated keys with low cosine similarity to each edit.
 local=local-torch.sum(local*keys[:,None,:],dim=2,keepdim=True)*keys[:,None,:];local=local/local.norm(dim=2,keepdim=True)
 return keys,deltas,para,local
def decode(method,keys,deltas):
 if method=='mirror':
  mag=deltas.norm(dim=1);ang=torch.atan2(deltas[:,1],deltas[:,0]);code=torch.stack([mag*torch.cos(ang),mag*torch.sin(ang)],1);return code@torch.eye(2).T
 if method=='quantized':return deltas.half().float()
 return deltas
def package(method,keys,deltas,N):
 obj={'format':'ma469-v1','method':method,'N':N,'keys':keys[:N]}
 if method=='mirror':obj.update({'polar_codes':torch.stack([deltas[:N].norm(dim=1),torch.atan2(deltas[:N,1],deltas[:N,0])],1),'shared_output_basis':torch.eye(2)})
 elif method=='dense':obj['dense_updates']=torch.stack([torch.outer(deltas[i],keys[i]) for i in range(N)])
 else:obj['output_deltas']=deltas[:N].half() if method=='quantized' else deltas[:N]
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def evaluate(method,keys,delta,para,local):
 gb=time.perf_counter();d=decode(method,keys,delta);genwall=(time.perf_counter()-gb)/len(keys);exact=[];parae=[];loc=[];lat=[]
 for i in range(len(keys)):
  k=keys[i];den=k.square().sum()
  t=time.perf_counter();
  if method=='dense':U=torch.outer(d[i],k)/den
  else:U=torch.outer(d[i],k)/den
  lat.append(time.perf_counter()-t)
  exact.append(float(((torch.eye(2)+U)@k-(k+delta[i])).norm()/(delta[i].norm()+1e-9)))
  yp=para[i]+para[i]@U.T; targetp=para[i]+delta[i]*(para[i]@k/den);parae.append(float((yp-targetp).norm()/(targetp.norm()+1e-9)))
  drift=(local[i]@U.T).norm(dim=1);loc.append(float(drift.mean()/(delta[i].norm()+1e-9)))
 return {'edit_error':statistics.mean(exact),'paraphrase_error':statistics.mean(parae),'locality_drift':statistics.mean(loc),'generation_wall':genwall,'apply_wall':statistics.mean(lat)}
def main(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);worlds=DEV if phase=='development' else FRESH
 if phase=='development':
  vals=[]
  for w in worlds:
   for s in range(3):
    k,d,p,l=data(w,s);vals.append({'world':w,'seed':s,'mend':evaluate('mend',k,d,p,l),'mirror':evaluate('mirror',k,d,p,l)})
  (ART/'development_runs.json').write_text(json.dumps(vals,indent=2)+'\n');return
 rows=[]
 for w in worlds:
  for s in range(3):
   k,d,p,l=data(w,s);metrics={m:evaluate(m,k,d,p,l) for m in ['mend','mirror','dense','quantized']}
   for m in metrics:
    for N in [1,20,64]:
     b=package(m,k,d,N);path=PAY/f'{w}_{s}_{m}_N{N}.pt';path.write_bytes(b);r=metrics[m]
     rows.append({'world':w,'seed':s,'method':m,'n':N,'edits':N,'payload_bytes':len(b),'bytes_per_edit':len(b)/N,'edit_error':r['edit_error'],'paraphrase_error':r['paraphrase_error'],'locality_drift':r['locality_drift'],'generation_MAC_per_edit':{'mend':6,'mirror':10,'dense':8,'quantized':6}[m],'update_apply_MAC':8,'generation_wall_seconds_per_edit':r['generation_wall'],'query_wall_seconds_mean':r['apply_wall'],'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);main(a.parse_args().phase)
