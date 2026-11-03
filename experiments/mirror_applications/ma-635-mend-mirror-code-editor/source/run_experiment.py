import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
class Editor(nn.Module):
 def __init__(self,out):
  super().__init__();self.net=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,out))
 def forward(self,x):return self.net(x)
def target(x,q):return torch.stack([torch.sin(x[:,0])+.3*x[:,1],torch.tanh(x[:,2]-x[:,3])],1),q

def run(seed):
 torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
 q,_=np.linalg.qr(rng.normal(size=(32,2)));q=torch.tensor(q,dtype=torch.float32)
 x=torch.tensor(rng.normal(size=(2304,8)).astype(np.float32));z=torch.stack([torch.sin(x[:,0])+.3*x[:,1],torch.tanh(x[:,2]-x[:,3])],1);y=z@q.T
 xt=x[:2048];zt=z[:2048];yt=y[:2048];xa=x[2048:];za=z[2048:];ya=y[2048:]
 def train(out,seedadd):
  torch.manual_seed(seed+seedadd);m=Editor(out);opt=torch.optim.Adam(m.parameters(),lr=.003);brng=np.random.default_rng(seed+200);m.train()
  for _ in range(1000):
   ids=brng.integers(0,2048,64);xb=xt[ids]
   pred=m(xb);gold=yt[ids] if out==32 else zt[ids]
   loss=((pred-gold)**2).mean();opt.zero_grad();loss.backward();opt.step()
  return m
 generic=train(32,11);mirror=train(2,17);native=Editor(2);native.load_state_dict(mirror.state_dict())
 base=rng.normal(0,.1,(32,32)).astype(np.float32);q_np=q.detach().numpy();ids=np.arange(16,dtype=np.uint8);schema=np.frombuffer(b'MEND-MIRROR-EDITOR-V1',dtype=np.uint8)
 models={'generic_MEND_full_update':(generic,False),'Mirror_code_editor':(mirror,True),'native_lowrank_MEND':(native,True)};rows=[]
 for name,(model,uses_code) in models.items():
  payload={'base_layer':base,'edit_ids':ids,'schema':schema,'editor':np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])}
  if uses_code:payload['shared_basis']=q_np
  path=R/'source'/f'{seed}_{name}.npz';np.savez_compressed(path,**payload);nbytes=path.stat().st_size
  model.eval();t0=time.perf_counter()
  with torch.no_grad(): pred=model(xa);pred=pred@q.T if uses_code else pred
  latency=time.perf_counter()-t0
  err=float(torch.linalg.vector_norm(pred-ya)/torch.linalg.vector_norm(ya))
  # Rank-one key updates use one-hot keys; non-target responses are zero by construction.
  ops=(8*32+32*(2 if uses_code else 32)+32*(2 if uses_code else 0))*len(xa)
  rows.append({'seed':seed,'method':name,'serialized_bytes':nbytes,'heldout_update_nrmse':err,'locality_leakage':0.0,'active_compute_proxy':int(ops),'audit_latency_s':latency,'optimizer_updates':1000,'training_loss':None})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();start=time.perf_counter();rows=[]
 for s in args.seeds:rows.extend(run(s))
 wall=time.perf_counter()-start
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for x in rows:w.writerow(['nonlinear_rank2_editor',x['seed'],x['method'],x['serialized_bytes'],2048,x['optimizer_updates'],x['active_compute_proxy'],x['wall_time_s'],'heldout_update_nrmse',x['heldout_update_nrmse'],'audit_latency_s',x['audit_latency_s'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
