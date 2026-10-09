import csv, hashlib, io, json, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; PAY=ART/'payloads'; D=512; N=32; DEV=(25500,25501); FRESH=(25510,25511,25512); SEEDS=(0,1,2); STEPS=1200
def gen(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+255);return torch.randn(N,D,generator=g),torch.randn(256,D,generator=g)
def psp(V,C):return (C.T@V)/C.shape[0]
def err(W,V,X):return float(torch.mean((X@W.T-X@V.T)**2).sqrt()/torch.mean((X@V.T)**2).sqrt())
def ser(o):b=io.BytesIO();torch.save(o,b);return b.getvalue()
def fit(V,rank,lr,steps,seed):
 torch.manual_seed(seed);E=torch.nn.Parameter(torch.randn(D,rank)*.01);C=torch.nn.Parameter(torch.randn(N,rank)*.01);opt=torch.optim.Adam([E,C],lr=lr)
 for _ in range(steps):
  opt.zero_grad();loss=((C@E.T-V)**2).mean();loss.backward();opt.step()
 return C.detach(),E.detach()
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True);rows=[]
 for w in DEV if phase=='development' else FRESH:
  for s in SEEDS:
   V,X=gen(w,s);C=torch.randint(0,2,(N,D),generator=torch.Generator().manual_seed(w+s+919)).float()*2-1;C/=D**.5;W=(C.T@V)/D
   for name,R,ctx in [('psp_random_unbind',W,C),('shared_tensor_no_task_code',V.sum(0,keepdim=True).expand(N,-1),torch.empty(0))]:
    for i in range(N):
     blob=ser({'shared':R if name=='psp_random_unbind' else V.sum(0),'contexts':ctx,'task':i,'metadata':{'method':name,'decoder':'C.T@V/D' if name.startswith('psp') else 'sum'}});path=PAY/f'{w}_{s}_{name}_{i}.pt';path.write_bytes(blob)
     rows.append({'world':w,'seed':s,'method':name,'task':i,'nrmse':err(R[i],V[i:i+1],X),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'train_seconds':0.,'updates':0})
   for lr in ([.001,.01,.1] if phase=='development' else [.01]):
    t=time.perf_counter();c,e=fit(V,8,lr,STEPS,w*17+s);dt=time.perf_counter()-t;R=c@e.T
    for i in range(N):
     blob=ser({'basis':e,'code':c[i],'task':i,'metadata':{'rank':8,'lr':lr}});path=PAY/f'{w}_{s}_mirror_rank8_lr{lr}_{i}.pt';path.write_bytes(blob)
     rows.append({'world':w,'seed':s,'method':f'mirror_rank8_lr{lr:g}','task':i,'nrmse':err(R[i],V[i:i+1],X),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'train_seconds':dt,'updates':STEPS})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
