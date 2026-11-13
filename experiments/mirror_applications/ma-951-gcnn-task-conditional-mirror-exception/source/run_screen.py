"""MA-951 exact C4 orbit representation with task phase and private exception."""
import csv,json,time,copy
from pathlib import Path
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
SEEDS=[95111,95113,95117];UPDATES=500;BATCH=64;T=3;C=8

def make_data(seed,n,train=False):
 rng=np.random.default_rng(seed);xs=[];ys=[]
 for _ in range(n):
  cls=int(rng.integers(0,2));ori=int(rng.integers(0,3 if train else 4));img=rng.normal(0,.015,(16,16)).astype('float32')
  img[6:10,6:10]+=0.25+0.45*cls
  img[1:4,6:10]+=0.9 # canonical top marker; rotate whole scene for C4 orientation
  img=np.rot90(img,ori).copy();xs.append(img[None]);ys.append([float(ori==0),float(ori==1),float(cls)])
 return torch.from_numpy(np.stack(xs)),torch.tensor(ys,dtype=torch.float32)
class Trunk(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(1,C,3,padding=1),nn.ReLU(),nn.Conv2d(C,C,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
 def forward(self,x):return self.net(x).flatten(1)
class TaskModel(nn.Module):
 def __init__(self,condition):
  super().__init__();self.condition=condition;self.trunk=Trunk()
  if condition=='native_heads':self.head=nn.Parameter(torch.empty(T,4,C));self.bias=nn.Parameter(torch.zeros(T))
  elif condition=='mirror_phase':self.shared=nn.Parameter(torch.empty(4,C));self.bias=nn.Parameter(torch.zeros(()));self.register_buffer('phase',torch.tensor([0,1,0],dtype=torch.long))
  elif condition=='mirror_private':self.shared=nn.Parameter(torch.empty(4,C));self.private=nn.Parameter(torch.empty(4,C));self.bias=nn.Parameter(torch.zeros(T));self.register_buffer('phase',torch.tensor([0,1,0],dtype=torch.long))
  elif condition=='ordinary_gate':self.shared=nn.Parameter(torch.empty(4,C));self.gate=nn.Parameter(torch.zeros(T,4));self.bias=nn.Parameter(torch.zeros(T))
  else:raise ValueError(condition)
  for p in [getattr(self,'head',None),getattr(self,'shared',None),getattr(self,'private',None)]:
   if isinstance(p,nn.Parameter):nn.init.xavier_uniform_(p)
 def features(self,x):
  return torch.stack([self.trunk(torch.rot90(x,k,(-2,-1))) for k in range(4)],dim=1)
 def forward(self,x):
  z=self.features(x)
  if self.condition=='native_heads':return (z[:,None]*self.head[None]).sum((2,3))+self.bias
  if self.condition=='mirror_phase':
   return torch.stack([(torch.roll(z,shifts=int(self.phase[t]),dims=1)*self.shared).sum((1,2))+self.bias for t in range(T)],dim=1)
  if self.condition=='mirror_private':
   out=[(torch.roll(z,shifts=int(self.phase[t]),dims=1)*self.shared).sum((1,2))+self.bias[t] for t in [0,1]]
   out.append((z*(self.shared+self.private)).sum((1,2))+self.bias[2]);return torch.stack(out,1)
  return (z[:,None]*self.shared[None,None]*torch.sigmoid(self.gate)[None,:,:,None]).sum((2,3))+self.bias

def equiv_error(model,x):
 with torch.no_grad():
  z=model.features(x[:32]);errs=[]
  for h in range(1,4):errs.append((model.features(torch.rot90(x[:32],h,(-2,-1)))-torch.roll(z,shifts=-h,dims=1)).abs().max().item())
 return max(errs)
def fit_eval(cond,seed,xtr,ytr,xte,yte):
 torch.manual_seed(seed);m=TaskModel(cond);opt=torch.optim.Adam(m.parameters(),lr=.003);rng=torch.Generator().manual_seed(seed+51);t0=time.perf_counter()
 for _ in range(UPDATES):
  ix=torch.randint(len(xtr),(BATCH,),generator=rng);logits=m(xtr[ix]);loss=F.binary_cross_entropy_with_logits(logits,ytr[ix]);opt.zero_grad();loss.backward();opt.step()
 train_s=time.perf_counter()-t0;m.eval()
 payload=ROOT/'source'/f'{cond}_{seed}.pt';torch.save({'state_dict':m.state_dict(),'condition':cond,'metadata':{'tasks':['orientation0','orientation1','object_class'],'group':'C4'}},payload);nbytes=payload.stat().st_size
 clone=TaskModel(cond);clone.load_state_dict(torch.load(payload,map_location='cpu',weights_only=True)['state_dict']);clone.eval()
 with torch.no_grad():z=clone(xte);pred=(z>=0).float();acc=(pred==yte).float().mean(0).tolist();err=float((z-m(xte)).abs().max())
 eq=equiv_error(clone,xte);start=time.perf_counter()
 with torch.no_grad():
  for st in range(0,len(xte),BATCH):clone(xte[st:st+BATCH])
 infer_s=time.perf_counter()-start;fps=len(xte)/infer_s
 # Four rotated CNN views: two convs with 8 channels, 16x16 outputs.
 trunk_mac=4*(16*16*C*1*3*3+16*16*C*C*3*3)
 if cond=='native_heads':head_mac=T*4*C
 elif cond=='ordinary_gate':head_mac=T*4*C*2
 else:head_mac=T*4*C+(4*C if cond=='mirror_private' else 0)
 payload.unlink()
 row={'seed':seed,'condition':cond,'acc_task0_orbit':acc[0],'acc_task1_orbit':acc[1],'acc_task2_nonorbit':acc[2],'equivariance_max_abs':eq,'payload_bytes':nbytes,'train_examples':len(xtr),'test_examples':len(xte),'updates':UPDATES,'train_seconds':train_s,'inference_examples_per_second':fps,'mac_proxy_per_example':trunk_mac+head_mac,'serialization_replay_max_abs':err}
 return row

def main():
 rows=[]
 for seed in SEEDS:
  xtr,ytr=make_data(seed,512,True);xte,yte=make_data(seed+1000,256,False)
  for cond in ['native_heads','mirror_phase','mirror_private','ordinary_gate']:
   row=fit_eval(cond,seed,xtr,ytr,xte,yte);rows.append(row);print(json.dumps(row),flush=True)
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'source/results.json').write_text(json.dumps({'results':rows,'seeds':SEEDS,'train_orientation_ids':[0,1,2],'heldout_orientation_id':3,'device':'cpu','torch':torch.__version__,'task_relation':'task0 and task1 are exact C4 orbit copies; task2 is object class'},indent=2)+'\n')
 print('max equivariance',max(r['equivariance_max_abs'] for r in rows),'max replay',max(r['serialization_replay_max_abs'] for r in rows))
if __name__=='__main__':main()
