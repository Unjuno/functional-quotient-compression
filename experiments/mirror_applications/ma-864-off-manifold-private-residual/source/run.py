"""Shared task-delta basis plus private rank-1 classifier residual."""
import argparse,io,json,random,time
import numpy as np,torch
from torch import nn
from torch.func import functional_call
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
WIDTH=32;ANGLES=[0,45,90,135,180,225,270];BASE_EPOCHS=40;SRC_EPOCHS=15;UPDATES=100;SUPPORT=32;BATCH=64

def data():
 d=load_digits();ims=d.images.astype(np.float32)/16.;y=d.target.astype(np.int64);x=np.stack([[rotate(im,a,reshape=False,order=1,mode='constant',cval=0,prefilter=False).reshape(-1) for a in ANGLES] for im in ims]);ids=np.arange(len(y));tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=y);va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=y[rest]);return (torch.tensor(x[tr]),torch.tensor(y[tr])),(torch.tensor(x[va]),torch.tensor(y[va])),(torch.tensor(x[te]),torch.tensor(y[te]))
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))
def train_base(seed,lr,epochs=BASE_EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+31);u=e=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   dom=torch.randint(0,6,(len(ids),),generator=g);loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();u+=1;e+=len(ids)
  sch.step()
 return m,u,e
def train_domain(base,seed,lr,dom,epochs=SRC_EPOCHS):
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
def build(seed,base_lr):
 base,bu,be=train_base(seed,base_lr);models=[];su=se=0
 for j in range(6):m,u,e=train_domain(base,seed,base_lr,j);models.append(m);su+=u;se+=e
 b=flat(base);d=torch.stack([flat(m)-b for m in models]);_,sv,vh=torch.linalg.svd(d,full_matrices=False);B=vh*sv[:,None]/np.sqrt(5);return base,b,B,sv,bu,be,su,se
def sample_support(seed,split):
 x,y=split;ids=np.random.default_rng(seed).choice(len(y),SUPPORT,replace=False);return x[ids,6],y[ids]
def eval_state(m,state,x,y):
 with torch.no_grad():z=functional_call(m,state,(x,));return float((z.argmax(1)==y).float().mean()),float(nn.functional.cross_entropy(z,y))
def fit_code(m,b,B,sx,sy,lr,steps=UPDATES):
 c=torch.zeros(B.shape[0],requires_grad=True);o=torch.optim.Adam([c],lr=lr)
 for _ in range(steps):theta=b+c@B;loss=nn.functional.cross_entropy(functional_call(m,unflat(m,theta),(sx,)),sy);o.zero_grad();loss.backward();o.step()
 return c.detach()
def fit_private(m,b,B,sx,sy,lr,steps=UPDATES):
 c=torch.zeros(B.shape[0],requires_grad=True);A=torch.zeros(10,1,requires_grad=True);V=torch.randn(1,WIDTH)*.01;V=V.requires_grad_();o=torch.optim.Adam([c,A,V],lr=lr)
 for _ in range(steps):
  p=unflat(m,b+c@B);p['fc2.weight']=p['fc2.weight']+A@V;loss=nn.functional.cross_entropy(functional_call(m,p,(sx,)),sy);o.zero_grad();loss.backward();o.step()
 return c.detach(),A.detach(),V.detach()
def fit_full(m,sx,sy,lr,steps=UPDATES):
 cp=Net();cp.load_state_dict(m.state_dict());o=torch.optim.Adam(cp.parameters(),lr=lr)
 for _ in range(steps):loss=nn.functional.cross_entropy(cp(sx),sy);o.zero_grad();loss.backward();o.step()
 return cp
def serialize(obj):
 bb=io.BytesIO();torch.save(obj,bb);return len(bb.getvalue()),bb.getvalue()
def payload(m,b,B,c,private=None):
 d={'base':m.state_dict(),'basis':B.detach().clone().cpu(),'code':c.detach().clone().cpu(),'architecture':'64-32-10-ReLU','normalization':'digits/16'}
 if private is not None:d['private_lora_fc2']={'A':private[0].cpu(),'B':private[1].cpu()}
 bb=io.BytesIO();torch.save(d,bb);return len(bb.getvalue()),bb.getvalue()
def run(seed,settings):
 base_lr,lr3,lr6,lrr,lrp,lrf=settings;torch.set_num_threads(1);train,_,test=data();start=time.perf_counter();m,b,B,sv,bu,be,su,se=build(seed,base_lr);sx,sy=sample_support(seed+777,train);tx,ty=test[0][:,6],test[1]
 B3=B[:3];B6=B[:6];g=torch.Generator().manual_seed(seed+991);q,_=torch.linalg.qr(torch.randn((len(b),3),generator=g));Br=q.T*sv[:3,None]/np.sqrt(5)
 c3=fit_code(m,b,B3,sx,sy,lr3);a3=unflat(m,b+c3@B3);acc3,nll3=eval_state(m,a3,tx,ty)
 c6=fit_code(m,b,B6,sx,sy,lr6);acc6,nll6=eval_state(m,unflat(m,b+c6@B6),tx,ty)
 cr=fit_code(m,b,Br,sx,sy,lrr);accr,nllr=eval_state(m,unflat(m,b+cr@Br),tx,ty)
 cp,pa,pv=fit_private(m,b,B3,sx,sy,lrp);pp=unflat(m,b+cp@B3);pp['fc2.weight']=pp['fc2.weight']+pa@pv;accp,nllp=eval_state(m,pp,tx,ty)
 full=fit_full(m,sx,sy,lrf);af,nllf=eval_state(full,full.state_dict(),tx,ty);baseacc,basenll=eval_state(m,m.state_dict(),tx,ty)
 size3,_=payload(m,b,B3,c3);size6,_=payload(m,b,B6,c6);sizer,_=payload(m,b,Br,cr);sizep,pblob=payload(m,b,B3,cp,(pa,pv));fullsize,_=serialize({'state':full.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});basisonly,_=serialize({'basis':B3.detach().clone().cpu()});basebytes,_=serialize({'state':m.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});loaded=torch.load(io.BytesIO(pblob),map_location='cpu',weights_only=False);lm=Net();lm.load_state_dict(loaded['base']);ltheta=flat(lm)+loaded['code']@loaded['basis'];lp=unflat(lm,ltheta);lp['fc2.weight']=lp['fc2.weight']+loaded['private_lora_fc2']['A']@loaded['private_lora_fc2']['B'];ldiff=float((functional_call(m,pp,(tx[:64],))-functional_call(lm,lp,(tx[:64],))).abs().max())
 # Identity Mirror chart on native coefficients: decode the exact same c.
 mdiff=float((functional_call(m,a3,(tx[:64],))-functional_call(m,unflat(m,b+c3.clone()@B3),(tx[:64],))).abs().max())
 return {'seed':seed,'settings':settings,'base_accuracy':baseacc,'base_nll':basenll,'rank3_accuracy':acc3,'rank3_nll':nll3,'rank6_accuracy':acc6,'rank6_nll':nll6,'random3_accuracy':accr,'random3_nll':nllr,'private_rank1_accuracy':accp,'private_rank1_nll':nllp,'full_accuracy':af,'full_nll':nllf,'rank3_payload_bytes':size3,'rank6_payload_bytes':size6,'random3_payload_bytes':sizer,'private_payload_bytes':sizep,'full_model_bytes':fullsize,'base_bytes':basebytes,'basis3_bytes':len(payload(m,b,B3,c3)[1])-len(payload(m,b,torch.empty(0),torch.empty(0))[1]),'base_updates':bu,'base_examples':be,'source_updates':su,'source_examples':se,'support_updates':UPDATES,'support_examples':SUPPORT,'reconstruction_macs':6*len(b),'private_residual_macs':10*WIDTH,'query_macs':len(ty)*(64*WIDTH+WIDTH*10),'wall_time_s':time.perf_counter()-start,'native_mirror_max_logit_diff':mdiff,'payload_reload_max_logit_diff':ldiff}
def tune():
 train,dev,_=data();base_lr=.1;base,b,B,sv,*_=build(73,base_lr);sx,sy=sample_support(773,train);x,y=dev[0][:,6],dev[1];q,_=torch.linalg.qr(torch.randn((len(b),3),generator=torch.Generator().manual_seed(977)));Br=q.T*sv[:3,None]/np.sqrt(5);choices=[.01,.1,.5];best={k:(1e9,None) for k in ['r3','r6','random','private','full']}
 for lr in choices:
  for key,bb in [('r3',B[:3]),('r6',B[:6]),('random',Br)]:
   c=fit_code(base,b,bb,sx,sy,lr);_,nll=eval_state(base,unflat(base,b+c@bb),x,y)
   if nll<best[key][0]:best[key]=(nll,lr)
  c,A,V=fit_private(base,b,B[:3],sx,sy,lr);p=unflat(base,b+c@B[:3]);p['fc2.weight']=p['fc2.weight']+A@V;_,nll=eval_state(base,p,x,y)
  if nll<best['private'][0]:best['private']=(nll,lr)
  f=fit_full(base,sx,sy,lr);_,nll=eval_state(f,f.state_dict(),x,y)
  if nll<best['full'][0]:best['full']=(nll,lr)
 return (base_lr,best['r3'][1],best['r6'][1],best['random'][1],best['private'][1],best['full'][1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked',nargs=6,type=float);a=p.parse_args();torch.set_num_threads(1);settings=tune() if a.locked is None else tuple(a.locked);seeds=[73] if a.development else ([311,331,347] if a.fresh else [73,311,331,347]);print(json.dumps({'selected_settings':settings,'runs':[run(s,settings) for s in seeds]},indent=2))
if __name__=='__main__':main()
