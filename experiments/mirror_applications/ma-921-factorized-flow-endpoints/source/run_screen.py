import json, random, time, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

class Net(nn.Module):
 def __init__(self,mode):
  super().__init__(); self.mode=mode
  self.net=nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2))
  self.start=nn.Embedding(8,2) if mode!='native' else None
  self.end=nn.Embedding(8,2) if mode!='native' else None
 def forward(self,x,i,j):
  if self.mode=='native': z=torch.stack((i.float()/7,j.float()/7),-1)
  else: z=self.start(i)+self.end(j)
  return self.net(torch.cat((x,z),-1))

def expA(A,d):
 ev,Q=torch.linalg.eig(A); D=torch.diag_embed(torch.exp(ev[None,:]*d[:,None])); return (Q[None]@D@torch.linalg.inv(Q)[None]).real

def truth(x,A,i,j):
 d=(j-i).float()/7
 return (expA(A,d)@x.unsqueeze(-1)).squeeze(-1)

worlds=[(92121,[[0.,-.8],[.8,-.08]]),(92123,[[-.1,-1.],[.7,-.15]]),(92129,[[.1,-.9],[1.1,-.2]])]
rows=[]
for seed,A0 in worlds:
 A=torch.tensor(A0,dtype=torch.float32)
 pairs=[(i,j) for i in range(8) for j in range(i+1,8)]
 chain={(i,i+1) for i in range(7)}
 held=[p for p in pairs if p not in chain and (p[0]*3+p[1]*5)%5==0]
 train=[p for p in pairs if p not in held]
 # Assert every time endpoint has training support.
 assert set(range(8))=={q for p in train for q in p}
 for mode in ['native','mirror','ordinary_endpoint_embedding']:
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); data_rng=torch.Generator().manual_seed(seed+1000)
  actual='mirror' if mode=='ordinary_endpoint_embedding' else mode
  model=Net(actual); opt=torch.optim.Adam(model.parameters(),lr=.002); t0=time.perf_counter()
  for _ in range(600):
   p=random.choice(train); n=128; x=torch.randn(n,2,generator=data_rng); i=torch.full((n,),p[0]); j=torch.full((n,),p[1]); y=truth(x,A,i,j)
   loss=((model(x,i,j)-y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
  seconds=time.perf_counter()-t0
  with torch.no_grad():
   errors=[]
   for i0,j0 in held:
    x=torch.randn(512,2,generator=data_rng); ii=torch.full((512,),i0); jj=torch.full((512,),j0); errors.append(((model(x,ii,jj)-truth(x,A,ii,jj))**2).mean().item())
   map_mse=sum(errors)/len(errors)
   comp=[]
   for i0 in range(6):
    for j0 in range(i0+1,7):
     for k0 in range(j0+1,8):
      x=torch.randn(16,2,generator=data_rng); ii=torch.full((16,),i0); jj=torch.full((16,),j0); kk=torch.full((16,),k0)
      comp.append(((model(model(x,ii,jj),jj,kk)-truth(x,A,ii,kk))**2).mean().item())
   comp_mse=sum(comp)/len(comp)
  path=ROOT/'source'/'payload.pt'; torch.save(model.state_dict(),path); size=path.stat().st_size
  clone=Net(actual); clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
  with torch.no_grad():
   rx=torch.randn(4,2,generator=data_rng); ri=torch.tensor([0,1,2,3]); rj=torch.tensor([1,2,3,4]); replay=float((clone(rx,ri,rj)-model(rx,ri,rj)).abs().max())
  # Deterministic state serialization size; transient checkpoint removed after measuring.
  path.unlink(); mac=4*32+32*32+32*2
  rows.append({'seed':seed,'condition':mode,'map_mse':map_mse,'composition_mse':comp_mse,'serialized_bytes':size,'mac_proxy':mac,'train_seconds':seconds,'roundtrip_max_abs':replay,'heldout_pairs':len(held)})
with (ROOT/'source'/'results.json').open('w') as f: json.dump({'results':rows,'device':'cpu','worlds':worlds},f,indent=2)
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(rows,indent=2))
