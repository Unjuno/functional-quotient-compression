"""Common-task coordinate tournament on four rotated Digits tasks."""
import argparse,io,json,random,time
import numpy as np,torch
from torch import nn
from torch.func import functional_call
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
WIDTH=32;BASE_EPOCHS=40;TASK_EPOCHS=20;BATCH=64

def data():
 d=load_digits();im=d.images.astype(np.float32)/16.;y=d.target.astype(np.int64);x=np.stack([im,np.rot90(im,1,axes=(1,2)),np.rot90(im,2,axes=(1,2)),np.rot90(im,3,axes=(1,2))],axis=1).reshape(len(y),4,64);ids=np.arange(len(y));tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=y);va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=y[rest]);return (torch.tensor(x[tr]),torch.tensor(y[tr])),(torch.tensor(x[va]),torch.tensor(y[va])),(torch.tensor(x[te]),torch.tensor(y[te]))
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))
def train_base(seed,lr,epochs=BASE_EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+19);u=e=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   dom=torch.randint(0,4,(len(ids),),generator=g);loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();u+=1;e+=len(ids)
  sch.step()
 return m,u,e
def train_task(base,seed,lr,dom,epochs=TASK_EPOCHS):
 (x,y),_,_=data();m=Net();m.load_state_dict(base.state_dict());o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+101+dom);u=e=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();u+=1;e+=len(ids)
  sch.step()
 return m,u,e
def flat(m):return torch.cat([p.detach().flatten() for p in m.parameters()])
def unflat(m,v):
 out={};i=0
 for n,p in m.named_parameters():out[n]=v[i:i+p.numel()].reshape(p.shape);i+=p.numel()
 return out
def serialize(o):b=io.BytesIO();torch.save(o,b);return len(b.getvalue()),b.getvalue()
def eval_by_task(m,split):
 x,y=split;out=[]
 with torch.no_grad():
  for i in range(4):z=m(x[:,i]);out.append((float((z.argmax(1)==y).float().mean()),float(nn.functional.cross_entropy(z,y))))
 return out
def eval_recon(base,vecs,split):
 x,y=split;out=[]
 with torch.no_grad():
  for i,v in enumerate(vecs):
   z=functional_call(base,unflat(base,v),(x[:,i],));out.append((float((z.argmax(1)==y).float().mean()),float(nn.functional.cross_entropy(z,y))))
 return out
def run(seed,base_lr,task_lr):
 torch.set_num_threads(1);train,val,test=data();st=time.perf_counter();base,bu,be=train_base(seed,base_lr);experts=[];tu=te=0
 for i in range(4):m,u,e=train_task(base,seed,task_lr,i);experts.append(m);tu+=u;te+=e
 b=flat(base);D=torch.stack([flat(m)-b for m in experts]);_,sv,vh=torch.linalg.svd(D,full_matrices=False);q,_=torch.linalg.qr(torch.randn((len(b),4),generator=torch.Generator().manual_seed(seed+701)));R=q.T;fullvec=[flat(m) for m in experts];indbytes,_=serialize({'models':[m.state_dict() for m in experts],'architecture':'64-32-10-ReLU','normalization':'digits/16'});taskbytes,_=serialize({'base':base.state_dict(),'task_deltas':D.detach().clone(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});soup=Net();soup.load_state_dict({k:sum(m.state_dict()[k]/4 for m in experts) for k in experts[0].state_dict()});soupmetrics=eval_by_task(soup,test);soupbytes,_=serialize({'state':soup.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});expertmetrics=eval_by_task(experts[0],test) # per-task experts evaluated below
 expert_by_task=[]
 with torch.no_grad():
  for i,m in enumerate(experts):z=m(test[0][:,i]);expert_by_task.append((float((z.argmax(1)==test[1]).float().mean()),float(nn.functional.cross_entropy(z,test[1]))))
 ranks={};P=len(b)
 for r in range(1,5):
  sb=vh[:r].clone();sc=D@sb.T;svb=R[:r].clone();rc=D@svb.T
  smodels=[b+sc[i]@sb for i in range(4)];rmodels=[b+rc[i]@svb for i in range(4)];smetrics=eval_recon(base,smodels,test);rmetrics=eval_recon(base,rmodels,test)
  sobj={'base':base.state_dict(),'basis':sb.clone(),'codes':sc.clone(),'rank':r,'architecture':'64-32-10-ReLU','normalization':'digits/16'};robj={'base':base.state_dict(),'basis':svb.clone(),'codes':rc.clone(),'rank':r,'architecture':'64-32-10-ReLU','normalization':'digits/16'};sbytes,sblob=serialize(sobj);rbytes,rblob=serialize(robj)
  # Explicit native/Mirror decoder identity check.
  mdiff=0.
  with torch.no_grad():
   for i in range(4):
    vecn=b+sc[i]@sb;vecm=b+sc[i].clone()@sb;xn=test[0][:32,i];zn=functional_call(base,unflat(base,vecn),(xn,));zm=functional_call(base,unflat(base,vecm),(xn,));mdiff=max(mdiff,float((zn-zm).abs().max()))
  ranks[str(r)]={'svd_metrics':smetrics,'random_metrics':rmetrics,'svd_payload_bytes':sbytes,'random_payload_bytes':rbytes,'svd_decode_macs':r*P*4,'random_decode_macs':r*P*4,'mirror_native_max_logit_difference':mdiff,'materialized_svd_bytes':serialize({'models':[unflat(base,v) for v in smodels],'architecture':'64-32-10-ReLU','normalization':'digits/16'})[0]}
 total=time.perf_counter()-st
 return {'seed':seed,'base_lr':base_lr,'task_lr':task_lr,'independent_task_metrics':expert_by_task,'model_soup_metrics':soupmetrics,'model_soup_bytes':soupbytes,'ranks':ranks,'independent_bank_bytes':indbytes,'full_task_vector_bytes':taskbytes,'base_bytes':serialize({'state':base.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'})[0],'base_updates':bu,'base_examples':be,'task_updates':tu,'task_examples':te,'query_macs_all_tasks':4*len(test[1])*(64*WIDTH+WIDTH*10),'training_wall_time_s':total}
def tune():
 _,val,_=data();best=None
 for blr in (.03,.1):
  for tlr in (.03,.1):
   base,_,_=train_base(83,blr);mods=[train_task(base,83,tlr,i)[0] for i in range(4)];scores=[]
   with torch.no_grad():
    for i,m in enumerate(mods):scores.append(float(nn.functional.cross_entropy(m(val[0][:,i]),val[1])))
   score=sum(scores)/4
   if best is None or score<best[0]:best=(score,blr,tlr)
 return best[1],best[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked',nargs=2,type=float);a=p.parse_args();torch.set_num_threads(1);settings=tune() if a.locked is None else tuple(a.locked);seeds=[83] if a.development else ([401,419,433] if a.fresh else [83,401,419,433]);print(json.dumps({'selected_settings':settings,'runs':[run(s,*settings) for s in seeds]},indent=2))
if __name__=='__main__':main()
