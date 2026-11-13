import json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'source' / 'results.json'
DEVICE = 'cpu'
torch.set_num_threads(1)

class MapNet(nn.Module):
    def __init__(self, view=False):
        super().__init__()
        self.view = view
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
    def forward(self, x, s, t):
        times = torch.stack((s, t-s), -1) if self.view else torch.stack((s,t), -1)
        return self.net(torch.cat((x, times), -1))

def matrix_exp(A, d):
    # Batched exact 2x2 matrix exponential via eigendecomposition, stable for these small real systems.
    ev, Q = torch.linalg.eig(A)
    vals = torch.exp(ev[None,:] * d[:,None])
    inv = torch.linalg.inv(Q)
    E = Q[None,:,:] @ torch.diag_embed(vals) @ inv[None,:,:]
    return E.real

def target(x, A, s, t):
    return torch.bmm(matrix_exp(A, t-s), x.unsqueeze(-1)).squeeze(-1)

def train_one(seed, A_np, view):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    A=torch.tensor(A_np,dtype=torch.float32)
    model=MapNet(view).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=0.001)
    start=time.perf_counter()
    for _ in range(600):
        n=256; x=torch.randn(n,2); s=torch.rand(n); t=s+(1-s)*torch.rand(n)
        y=target(x,A,s,t)
        pred=model(x,s,t); loss=((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    train_sec=time.perf_counter()-start
    with torch.no_grad():
        n=4096; x=torch.randn(n,2); s=torch.rand(n); t=s+(1-s)*torch.rand(n)
        mse=((model(x,s,t)-target(x,A,s,t))**2).mean().item()
        a=torch.rand(n); b=a+(1-a)*torch.rand(n); c=b+(1-b)*torch.rand(n); x0=torch.randn(n,2)
        composed=model(model(x0,a,b),b,c); direct=model(x0,a,c)
        comp=((composed-direct)**2).mean().item()
    p=ROOT/'source'/(f'{seed}_'+('mirror' if view else 'native')+'.pt')
    torch.save(model.state_dict(),p)
    nbytes=p.stat().st_size
    # Verify serialized state reproduces a deterministic held-out output exactly.
    clone=MapNet(view).to(DEVICE); clone.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    with torch.no_grad():
        replay=torch.max(torch.abs(clone(x[:32],s[:32],t[:32])-model(x[:32],s[:32],t[:32]))).item()
    mac=4*64 + 64*64 + 64*2
    p.unlink()
    return {'map_mse':mse,'composition_mse':comp,'serialized_bytes':nbytes,'mac_proxy':mac,'train_seconds':train_sec,'replay_max_abs':replay}

worlds=[
 (91911,[[0.,-1.],[1.,-.1]]),
 (91913,[[-.2,-.7],[.9,-.05]]),
 (91917,[[.15,-1.2],[.8,-.25]])]
rows=[]
for seed,A in worlds:
 for name,view in [('native',False),('mirror_start_duration',True)]:
  rows.append({'seed':seed,'condition':name,**train_one(seed,A,view)})
OUT.write_text(json.dumps({'device':DEVICE,'torch':torch.__version__,'protocol':'MA-919 amendment 01','results':rows},indent=2)+'\n')
import csv
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['seed','condition','map_mse','composition_mse','serialized_bytes','mac_proxy','train_seconds','replay_max_abs','status','notes']); w.writeheader()
 for r in rows: w.writerow({**r,'status':'SCREENING','notes':'Fresh analytic linear-flow world'})
print(json.dumps(rows,indent=2))
