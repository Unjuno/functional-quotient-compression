#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,random,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';P=A/'payloads';D=8;DEV=[44900,44901];FRESH=[44910,44911,44912];SEEDS=[0,1,2]
def basis(w):
 g=torch.Generator().manual_seed(w+994400);q,_=torch.linalg.qr(torch.randn(D,2,generator=g));return q*.65
def sample(w,seed,fac,n,offset):
 g=torch.Generator().manual_seed(w*1000003+seed*991+fac[0]*73+fac[1]*193+offset+n);x=torch.randn(n,D,generator=g);y=x@(basis(w)@torch.tensor(fac,dtype=torch.float32));return x,y
def calibrate(w,seed):
 cols=[]
 for j in range(2):
  f=(1,0) if j==0 else (0,1);x,y=sample(w,seed,f,64,117+j);cols.append(torch.linalg.lstsq(x,y).solution)
 return torch.stack(cols,dim=1)
def err(pred,y):return (pred-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)
def infer_leo(zbasis,x,y,lr,steps=5):
 z=torch.zeros(2,requires_grad=True)
 for _ in range(steps):
  loss=(x@(zbasis@z)-y).square().mean();g=torch.autograd.grad(loss,z)[0];z=(z-lr*g).detach().requires_grad_(True)
 return z.detach()
def evaluate(w,seed,tid,method,decoder,lr):
 x,y=sample(w,seed,(1,1),8,3000+tid);qx,qy=sample(w,seed,(1,1),128,6000+tid);t=time.perf_counter()
 if method=='mirror':z=torch.linalg.lstsq(x@decoder,y).solution;pred=qx@(decoder@z)
 elif method=='leo':z=infer_leo(decoder,x,y,lr);pred=qx@(decoder@z)
 elif method=='taskvec':z=torch.ones(2);pred=qx@(decoder@z)
 else:z=torch.linalg.lstsq(x,y).solution;pred=qx@z
 wall=time.perf_counter()-t;return float(err(pred,qy)),z.detach(),wall

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);a=ap.parse_args();A.mkdir(exist_ok=True);P.mkdir(exist_ok=True)
 if a.phase=='development':
  scores={}
  for lr in [.001,.01,.05,.1,.3]:
   es=[]
   for w in DEV:
    for seed in SEEDS:
     dec=calibrate(w,seed)
     for tid in range(100,120):es.append(evaluate(w,seed,tid,'leo',dec,lr)[0])
   scores[str(lr)]=sum(es)/len(es)
  chosen=min(scores,key=scores.get);(A/'development_selection.json').write_text(json.dumps({'leo_lr':float(chosen),'scores':scores,'fresh_untouched':True},indent=2)+'\n');print(json.dumps({'selected_leo_lr':chosen,'scores':scores},indent=2));return
 lr=json.loads((A/'development_selection.json').read_text())['leo_lr'];rows=[]
 for w in FRESH:
  for seed in SEEDS:
   dec=calibrate(w,seed)
   # Independent-factor intervention check: each isolated concept must map to its own coordinate.
   intervention=[]
   for fac in [(1,0),(0,1)]:
    ix,iy=sample(w,seed,fac,8,881);intervention.append(torch.linalg.lstsq(ix@dec,iy).solution.tolist())
   per={m:[] for m in ['taskvec','leo','mirror','full']};codes={m:[] for m in per};walls={m:[] for m in per}
   for tid in range(3000,3020):
    for m in per:
     e,z,wall=evaluate(w,seed,tid,m,dec,lr);per[m].append(e);codes[m].append(z);walls[m].append(wall)
   for m in per:
    obj={'format':'ma449-v1','method':m,'decoder':dec,'task_ids':list(range(3000,3020)),'codes':torch.stack(codes[m]),'metadata':{'factor_pair':[1,1],'leo_lr':lr if m=='leo' else None}}
    if m=='taskvec':obj['codes']=torch.ones(20,2)
    if m=='full':obj={'format':'ma449-v1','method':'full','task_ids':list(range(3000,3020)),'weights':torch.stack(codes[m]),'metadata':{'factor_pair':[1,1]}}
    b=io.BytesIO();torch.save(obj,b);data=b.getvalue();p=P/f'{w}_{seed}_{m}_N20.pt';p.write_bytes(data)
    rows.append({'world':w,'seed':seed,'method':m,'heldout_combo':'1,1','nrmse_mean':sum(per[m])/20,'payload_bytes':len(data),'bytes_per_task':len(data)/20,'query_wall_seconds_mean':sum(walls[m])/20,'adaptation_mac':80 if m=='leo' else 16 if m=='mirror' else 0,'intervention_coordinates':json.dumps(intervention),'hash':hashlib.sha256(data).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with open(A/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows)},indent=2))
if __name__=='__main__':main()
