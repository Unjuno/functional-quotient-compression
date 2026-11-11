"""MA-857 jointly trained barycentric weight simplex on rotated sklearn Digits."""
import argparse, io, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ALPHAS=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]],dtype=np.float32)
EPOCHS=80; BATCH=64; WIDTH=32

def get_data():
 d=load_digits(); imgs=d.images.astype(np.float32)/16.; labels=d.target.astype(np.int64)
 x=np.stack([imgs,np.rot90(imgs,1,axes=(1,2)),np.rot90(imgs,2,axes=(1,2))],axis=1).reshape(len(labels),3,64)
 ids=np.arange(len(labels))
 tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=labels)
 va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=labels[rest])
 def expand(src):
  return torch.tensor(x[src].reshape(-1,64)),torch.tensor(np.repeat(labels[src],3)),torch.tensor(np.tile(np.arange(3),len(src)))
 return expand(tr),expand(va),expand(te)

class Net(nn.Module):
 def __init__(self): super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))

def mixed_batch(x,y,ids,alpha,g):
 n=len(ids); dom=torch.multinomial(torch.tensor(alpha,dtype=torch.float32),n,replacement=True,generator=g); idx=ids
 # Each source digit has three adjacent domain rows in the flattened dataset.
 src=idx//3; chosen=src*3+dom
 return x[chosen],y[chosen]

def train_simplex(seed,lr,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y,_),_,_=get_data();nets=[Net() for _ in range(3)];params=[p for n in nets for p in n.parameters()];opt=torch.optim.SGD(params,lr=lr,momentum=.9,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);g=torch.Generator().manual_seed(seed+97);updates=examples=0;st=time.perf_counter()
 for ep in range(epochs):
  order=torch.randperm(len(x)//3,generator=g)
  for src_ids in order.split(BATCH):
   if len(src_ids)<2:continue
   alpha=torch.distributions.Dirichlet(torch.ones(3)).sample().numpy(); ids=src_ids*3
   xb,yb=mixed_batch(x,y,ids,alpha,g)
   sd=[n.state_dict(keep_vars=True) for n in nets]; mix={k:sum(float(alpha[i])*sd[i][k] for i in range(3)) for k in sd[0]}
   loss=nn.functional.cross_entropy(functional_call(nets[0],mix,(xb,)),yb);opt.zero_grad();loss.backward();opt.step();updates+=1;examples+=len(yb)
  sched.step()
 return nets,updates,examples,time.perf_counter()-st

def train_one(seed,lr,domain=None,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y,d),_,_=get_data();m=Net();opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);g=torch.Generator().manual_seed(seed+97);updates=examples=0
 for ep in range(epochs):
  if domain is None: src=torch.randperm(len(x)//3,generator=g)
  else:
   # use one sample per source image with selected rotation
   mask=(d==domain);src=torch.nonzero(mask).flatten();src=src[torch.randperm(len(src),generator=g)]
  for ii in src.split(BATCH):
   if domain is None:
    # indices are source ids; sample domains uniformly per example
    dom=torch.randint(0,3,(len(ii),),generator=g);idx=ii*3+dom
   else:idx=ii
   loss=nn.functional.cross_entropy(m(x[idx]),y[idx]);opt.zero_grad();loss.backward();opt.step();updates+=1;examples+=len(ii)
  sched.step()
 return m,updates,examples

def eval_grid(nets,split):
 x,y,d=split;out=[]
 with torch.no_grad():
  states=[n.state_dict() for n in nets]
  for a in ALPHAS:
   mix={k:sum(float(a[i])*states[i][k] for i in range(3)) for k in states[0]}
   # Matching mixture distribution, with deterministic proportional sample counts.
   ix=[]
   for dom,w in enumerate(a):
    if w>0: ix.extend(torch.nonzero(d==dom).flatten().tolist())
   # Weighted sample selection avoids giving every domain equal mass when alpha is unequal.
   rng=np.random.default_rng(991); picked=[]
   for dom,w in enumerate(a):
    n=int(round(360*float(w)))
    options=torch.nonzero(d==dom).flatten().numpy()
    if n and len(options):picked.extend(rng.choice(options,n,replace=len(options)<n).tolist())
   ix=torch.tensor(picked,dtype=torch.long);z=functional_call(nets[0],mix,(x[ix],));yy=y[ix]
   out.append((float((z.argmax(1)==yy).float().mean()),float(nn.functional.cross_entropy(z,yy))))
 return out

def serialize(obj):
 b=io.BytesIO();torch.save(obj,b);return len(b.getvalue()),b.getvalue()

def run(seed,lr):
 torch.set_num_threads(1);_,_,test=get_data();st=time.perf_counter();nets,u,e,t=train_simplex(seed,lr);metrics=eval_grid(nets,test)
 # Standard pooled comparator, equal updates.
 pooled,up,ex=train_one(seed+1000,lr,None); px,py,pd=test
 with torch.no_grad():
  zp=pooled(px);std_acc=float((zp.argmax(1)==py).float().mean());std_nll=float(nn.functional.cross_entropy(zp,py))
 # Three independently trained vertex models; each trained on own rotation.
 ind_start=time.perf_counter();indep=[train_one(seed+2000+i,lr,i)[0] for i in range(3)];ind_wall=time.perf_counter()-ind_start
 # Each uses separate task data; exact model soup baseline.
 soup=Net(); soup.load_state_dict({k:sum(m.state_dict()[k]/3 for m in indep) for k in indep[0].state_dict()})
 dom_acc=[]
 with torch.no_grad():
  for i,m in enumerate(indep):
   ix=pd==i;dom_acc.append(float((m(px[ix]).argmax(1)==py[ix]).float().mean()))
 soup_acc=float((soup(px).argmax(1)==py).float().mean())
 payload={'vertices':[{k:v.cpu() for k,v in n.state_dict().items()} for n in nets],'codes':ALPHAS[:,:2].copy(),'architecture':'64-32-10-ReLU','normalization':'digits/16','third_coordinate':'1-a0-a1'};size,blob=serialize(payload);loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
 ms=[Net() for _ in range(3)]
 for m,s in zip(ms,loaded['vertices']):m.load_state_dict(s)
 reload_metrics=eval_grid(ms,test)
 mirror_diff=0.
 with torch.no_grad():
  ss=[m.state_dict() for m in ms];xx=test[0][:32]
  for a in ALPHAS:
   m=2*a-1;ad=(m+1)/2;pn={k:sum(float(a[i])*ss[i][k] for i in range(3)) for k in ss[0]};pm={k:sum(float(ad[i])*ss[i][k] for i in range(3)) for k in ss[0]};mirror_diff=max(mirror_diff,float((functional_call(ms[0],pn,(xx,))-functional_call(ms[0],pm,(xx,))).abs().max()))
 mats=[{k:sum(float(a[i])*nets[i].state_dict()[k] for i in range(3)) for k in nets[0].state_dict()} for a in ALPHAS];matbytes,_=serialize({'models':mats,'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 stdbytes,_=serialize({'state':pooled.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 indbytes,_=serialize({'models':[m.state_dict() for m in indep],'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 return {'seed':seed,'lr':lr,'grid_metrics':metrics,'standard_accuracy':std_acc,'standard_nll':std_nll,'independent_vertex_accuracy':dom_acc,'soup_accuracy':soup_acc,'payload_bytes':size,'materialized_grid_bytes':matbytes,'standard_payload_bytes':stdbytes,'independent_bank_bytes':indbytes,'train_updates':u,'train_examples':e,'pooled_updates':up,'pooled_examples':ex,'train_wall_time_s':t,'independent_train_wall_time_s':ind_wall,'decode_macs':3*2410*10,'grid_query_macs':len(ALPHAS)*3*360*(64*WIDTH+WIDTH*10),'mirror_max_logit_diff':mirror_diff,'reload_max_accuracy_diff':max(abs(a[0]-b[0]) for a,b in zip(metrics,reload_metrics)),'wall_time_s':time.perf_counter()-st}

def tune():
 _,dev,_=get_data();best=None
 for lr in (.03,.1):
  models,*_=train_simplex(41,lr);m=eval_grid(models,dev);score=np.mean([x[1] for x in m])
  if best is None or score<best[0]:best=(score,lr)
 return best[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked-lr',type=float);a=p.parse_args();torch.set_num_threads(1);lr=a.locked_lr if a.locked_lr is not None else tune();seeds=[41] if a.development else ([131,149,167] if a.fresh else [41,131,149,167]);print(json.dumps({'selected_lr':lr,'grid':ALPHAS.tolist(),'runs':[run(s,lr) for s in seeds]},indent=2))
if __name__=='__main__':main()
