import csv,hashlib,io,json,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(2);M=4;D=20;H=64;C=4

def data(world,seed,ntr=4096,nte=2048):
 g=torch.Generator().manual_seed(world*100003+seed*7919+260);cent=torch.randn(C,D,generator=g)*1.8; y=torch.randint(C,(ntr+nte,),generator=g);x=cent[y]+torch.randn(ntr+nte,D,generator=g)*1.1;return x[:ntr],y[:ntr],x[ntr:],y[ntr:]
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(D,H);self.fc2=nn.Linear(H,C)
 def forward(self,x):return self.fc2(torch.tanh(self.fc1(x)))
class Batch(nn.Module):
 def __init__(self):super().__init__();self.w=nn.Parameter(torch.empty(H,D));self.b=nn.Parameter(torch.zeros(H));self.v=nn.Parameter(torch.empty(D,M));self.r=nn.Parameter(torch.empty(H,M));self.out=nn.Linear(H,C);nn.init.kaiming_uniform_(self.w,a=5**.5);nn.init.normal_(self.v,std=.1);nn.init.normal_(self.r,std=.1)
 def forward(self,x,i):
  h=torch.tanh((x*self.v[:,i])@self.w.T*self.r[:,i]+self.b);return self.out(h)
class Mirror(nn.Module):
 def __init__(self):super().__init__();self.base=nn.Linear(D,H);self.theta=nn.Parameter(torch.zeros(M));self.phase=nn.Parameter(torch.zeros(M,H));self.out=nn.Linear(H,C)
 def forward(self,x,i):
  h=self.base(x);a=self.theta[i];# fixed paired channel rotations across even/odd coordinates
  z=h.clone();u=h[:,0::2];v=h[:,1::2];co=torch.cos(a);si=torch.sin(a);z[:,0::2]=co*u-si*v;z[:,1::2]=si*u+co*v
  return self.out(torch.tanh(z+self.phase[i]))
def metrics(model,kind,x,y):
 logits=[]
 with torch.no_grad():
  for i in range(M): logits.append(model(x,i) if kind!='single' else model(x))
 p=torch.stack(logits).softmax(-1).mean(0);acc=float((p.argmax(-1)==y).float().mean());nll=float(-p[torch.arange(len(y)),y].clamp_min(1e-9).log().mean());conf,pred=p.max(-1);ece=0.
 for b in range(10):
  ix=(conf>=b/10)&(conf<(b+1)/10)
  if ix.any():ece+=float(ix.float().mean()*abs((pred[ix]==y[ix]).float().mean()-conf[ix].mean()))
 return acc,nll,ece

def train(kind,seed,steps=500):
 torch.manual_seed(seed)
 model=Net() if kind=='single' else Batch() if kind=='batchensemble' else Mirror();opt=torch.optim.Adam(model.parameters(),lr=.003);x,y,xt,yt=data(26000,seed)
 t=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad()
  if kind=='single':loss=nn.functional.cross_entropy(model(x),y)
  else:loss=sum(nn.functional.cross_entropy(model(x,i),y) for i in range(M))/M
  loss.backward();opt.step()
 return model,time.perf_counter()-t,xt,yt

def blob(model,kind):
 b=io.BytesIO();torch.save({'kind':kind,'state_dict':model.state_dict(),'members':M,'metadata':{'input':D,'hidden':H,'classes':C}},b);return b.getvalue()
def run(phase):
 rows=[];worlds=[26000,26001] if phase=='development' else [26010,26011,26012]
 for w in worlds:
  for s in [0,1,2]:
   for kind in ['single','batchensemble','mirror']:
    model,sec,_,_=train(kind,s,500);xt,yt,_,_=data(w,s);acc,nll,ece=metrics(model,kind,xt,yt);payload=blob(model,kind);path=ROOT/'artifacts'/f'{w}_{s}_{kind}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(payload);rows.append({'world':w,'seed':s,'method':kind,'accuracy':acc,'nll':nll,'ece':ece,'payload_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'train_seconds':sec,'inference_examples_per_second':len(yt)*M/max(sec,1e-9)})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=rows[0]);wri.writeheader();wri.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
