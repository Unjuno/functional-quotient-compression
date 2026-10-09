import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[48700,48701];FRESH=[48710,48711,48712];SEEDS=[0,1,2];D=32;M=64;N=128;TOPK=8;STEPS=[1,2,4,8]
g=torch.Generator().manual_seed(486001);DICT=torch.nn.functional.normalize(torch.randn(D,M,generator=g),dim=0)
def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);c=torch.zeros(N,M)
 for i in range(N):
  ids=torch.randperm(M,generator=g)[:4];c[i,ids]=torch.randn(4,generator=g)
 return c@DICT.T
def train():
 ys=torch.cat([world(w,s) for w in DEV for s in SEEDS]); Ws=torch.nn.Parameter(torch.stack([.5*DICT.T for _ in range(8)])); Ss=torch.nn.Parameter(torch.stack([torch.eye(M) for _ in range(8)])); th=torch.nn.Parameter(torch.full((8,M),.04));opt=torch.optim.Adam([Ws,Ss,th],lr=.01)
 for _ in range(300):
  opt.zero_grad();z=torch.zeros(len(ys),M)
  for i in range(8):
   u=z@Ss[i].T+ys@Ws[i].T;z=u.sign()*torch.relu(u.abs()-th[i].abs())
  loss=((z@DICT.T-ys)**2).mean()+.001*z.abs().mean();loss.backward();opt.step()
 torch.save({'W':Ws.detach(),'S':Ss.detach(),'threshold':th.detach()},ART/'lista.pt')
def encode(y,model,steps):
 z=torch.zeros(len(y),M);W=model['W'];S=model['S'];th=model['threshold']
 for i in range(steps):
  u=z@S[i].T+y@W[i].T;z=u.sign()*torch.relu(u.abs()-th[i].abs())
  if i==steps-1:
   ids=z.abs().topk(TOPK,dim=1).indices;vals=z.gather(1,ids);return ids.to(torch.uint8),vals

def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':train();print('trained LISTA on development worlds');return
 model=torch.load(ART/'lista.pt',weights_only=False);rows=[]
 for w in FRESH:
  for s in SEEDS:
   y=world(w,s)
   for method in ['omp','lista']:
    for steps in ([0] if method=='omp' else STEPS):
     if method=='omp':
      residual=y.clone();ii=[];vv=[]
      for _ in range(TOPK):
       score=residual@DICT;idx=score.abs().argmax(1);val=score.gather(1,idx[:,None]).squeeze(1);ii.append(idx);vv.append(val);residual-=val[:,None]*DICT[:,idx].T
      ids=torch.stack(ii,1).to(torch.uint8);vals=torch.stack(vv,1)
     else:ids,vals=encode(y,model,steps)
     recon=torch.zeros_like(y)
     for j in range(TOPK):recon+=vals[:,j,None]*DICT[:,ids[:,j].long()].T
     err=float((recon-y).norm()/(y.norm()+1e-12));obj={'method':method,'dictionary':DICT,'ids':ids,'values':vals,'N':N,'steps':steps}
     if method=='lista':obj['router']=model
     b=io.BytesIO();torch.save(obj,b);blob=b.getvalue();path=PAY/f'{w}_{s}_{method}_S{steps}.pt';path.write_bytes(blob)
     rows.append({'world':w,'seed':s,'method':method,'steps':steps,'normalized_rmse':err,'active_atoms':TOPK,'payload_bytes':len(blob),'bytes_per_function':len(blob)/N,'inference_MAC_proxy':0 if method=='omp' else steps*(M*M+M*D),'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
