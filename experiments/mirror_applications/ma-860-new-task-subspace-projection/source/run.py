"""Few-shot projection into a rank-3 task subspace on rotated Digits."""
import argparse,io,json,random,time
import numpy as np,torch
from torch import nn
from torch.func import functional_call
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
WIDTH=32;BASE_EPOCHS=40;SRC_EPOCHS=15;UPDATES=100;SUPPORT=32;BATCH=64

def data():
 d=load_digits();im=d.images.astype(np.float32)/16.;y=d.target.astype(np.int64);x=np.stack([im,np.rot90(im,1,axes=(1,2)),np.rot90(im,2,axes=(1,2)),np.rot90(im,3,axes=(1,2))],axis=1).reshape(len(y),4,64);ids=np.arange(len(y));tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=y);va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=y[rest]);return (torch.tensor(x[tr]),torch.tensor(y[tr])),(torch.tensor(x[va]),torch.tensor(y[va])),(torch.tensor(x[te]),torch.tensor(y[te]))
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))
def train_base(seed,lr,epochs=BASE_EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+31);steps=ex=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   dom=torch.randint(0,3,(len(ids),),generator=g);z=m(x[ids,dom]);loss=nn.functional.cross_entropy(z,y[ids]);o.zero_grad();loss.backward();o.step();steps+=1;ex+=len(ids)
  sch.step()
 return m,steps,ex
def train_domain(base,seed,lr,dom,epochs=SRC_EPOCHS):
 (x,y),_,_=data();m=Net();m.load_state_dict(base.state_dict());o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+101+dom);steps=ex=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();steps+=1;ex+=len(ids)
  sch.step()
 return m,steps,ex
def flat(m):return torch.cat([p.detach().flatten() for p in m.parameters()])
def layout(m):return [(n,tuple(p.shape),p.numel()) for n,p in m.named_parameters()]
def unflat(m,v):
 out={};i=0
 for n,p in m.named_parameters():out[n]=v[i:i+p.numel()].reshape(p.shape);i+=p.numel()
 return out
def build(seed,base_lr):
 base,bu,be=train_base(seed,base_lr);mods=[];su=se=0
 for dom in range(3):m,u,e=train_domain(base,seed,base_lr,dom);mods.append(m);su+=u;se+=e
 b=flat(base);deltas=torch.stack([flat(m)-b for m in mods]);_,sv,vh=torch.linalg.svd(deltas,full_matrices=False);basis=vh[:3]*sv[:3,None]/np.sqrt(2);return base,b,basis,sv,bu,be,su,se
def support_sample(seed,split):
 x,y=split;g=np.random.default_rng(seed);ids=g.choice(len(y),SUPPORT,replace=False);return x[ids,3],y[ids]
def eval_code(base,b,basis,code,x,y):
 theta=b+code@basis;params=unflat(base,theta);without=None
 with torch.no_grad():z=functional_call(base,params,(x,));return float((z.argmax(1)==y).float().mean()),float(nn.functional.cross_entropy(z,y))
def fit_code(base,b,basis,sx,sy,lr,steps=UPDATES):
 code=torch.zeros(basis.shape[0],requires_grad=True);opt=torch.optim.Adam([code],lr=lr)
 for _ in range(steps):
  theta=b+code@basis;z=functional_call(base,unflat(base,theta),(sx,));loss=nn.functional.cross_entropy(z,sy);opt.zero_grad();loss.backward();opt.step()
 return code.detach()
def fit_full(base,sx,sy,lr,steps=UPDATES):
 m=Net();m.load_state_dict(base.state_dict());o=torch.optim.Adam(m.parameters(),lr=lr)
 for _ in range(steps):loss=nn.functional.cross_entropy(m(sx),sy);o.zero_grad();loss.backward();o.step()
 return m
def serialize(v):bb=io.BytesIO();torch.save(v,bb);return len(bb.getvalue()),bb.getvalue()
def run(seed,base_lr,learn_lr,rand_lr,full_lr):
 torch.set_num_threads(1);train,dev,test=data();st=time.perf_counter();base,b,B,sv,bu,be,su,se=build(seed,base_lr);sx,sy=support_sample(seed+700,train);tx,ty=test[0][:,3],test[1]
 learned=fit_code(base,b,B,sx,sy,learn_lr);acc,nll=eval_code(base,b,B,learned,tx,ty)
 # Random rank-3 orthonormal directions scaled to learned PCA loading norms.
 gen=torch.Generator().manual_seed(seed+900);q,_=torch.linalg.qr(torch.randn((len(b),3),generator=gen));rb=(q.T*sv[:3,None]/np.sqrt(2));rcode=fit_code(base,b,rb,sx,sy,rand_lr);racc,rnll=eval_code(base,b,rb,rcode,tx,ty)
 full=fit_full(base,sx,sy,full_lr)
 with torch.no_grad():
  fz=full(tx);facc=float((fz.argmax(1)==ty).float().mean());fnll=float(nn.functional.cross_entropy(fz,ty))
 base_size,_=serialize({'state':base.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});basis_obj={'base':base.state_dict(),'basis':B.cpu(),'code':learned.cpu(),'architecture':'64-32-10-ReLU','normalization':'digits/16'};pack_size,pack_blob=serialize(basis_obj);full_size,_=serialize({'state':full.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'});loaded=torch.load(io.BytesIO(pack_blob),map_location='cpu',weights_only=False);ldiff=0.;logdiff=0.
 with torch.no_grad():
  zz=eval_code(base,b,B,learned,tx[:64],ty[:64]);lb=loaded['base'];lm=Net();lm.load_state_dict(lb);ll=eval_code(lm,flat(lm),loaded['basis'],loaded['code'],tx[:64],ty[:64]);ldiff=max(abs(zz[0]-ll[0]),abs(zz[1]-ll[1]));theta=b+learned@B;ref=functional_call(base,unflat(base,theta),(tx[:64],));ltheta=flat(lm)+loaded['code']@loaded['basis'];out=functional_call(lm,unflat(lm,ltheta),(tx[:64],));logdiff=float((ref-out).abs().max())
 # Mirror m is an identity view of native coefficients; deterministic logits must coincide.
 mdiff=0.;native=eval_code(base,b,B,learned,tx[:64],ty[:64]);mirror=eval_code(base,b,B,learned.clone(),tx[:64],ty[:64]);mdiff=max(abs(native[0]-mirror[0]),abs(native[1]-mirror[1]))
 return {'seed':seed,'base_lr':base_lr,'learned_code_lr':learn_lr,'random_code_lr':rand_lr,'full_lr':full_lr,'learned_accuracy':acc,'learned_nll':nll,'random_accuracy':racc,'random_nll':rnll,'full_accuracy':facc,'full_nll':fnll,'base_accuracy':eval_code(base,b,B,torch.zeros(3),tx,ty)[0],'base_bytes':base_size,'shared_base_basis_code_bytes':pack_size,'full_adapted_model_bytes':full_size,'basis_bytes':len(serialize({'basis':B.cpu()})[1]),'basis_singular_values':sv.tolist(),'source_updates':su,'source_examples':se,'base_updates':bu,'base_examples':be,'support_updates':UPDATES,'support_examples':SUPPORT,'reconstruction_macs':3*len(b),'query_macs':len(ty)*(64*WIDTH+WIDTH*10),'train_wall_time_s':time.perf_counter()-st,'mirror_native_metric_difference':mdiff,'payload_reload_metric_difference':ldiff,'payload_reload_max_logit_difference':logdiff}
def tune():
 train,dev,_=data();base_lr=.1;base,b,B,sv,*_=build(53,base_lr);sx,sy=support_sample(753,train);tx,ty=dev[0][:,3],dev[1];best={k:(1e9,None) for k in ['learn','random','full']};q,_=torch.linalg.qr(torch.randn((len(b),3),generator=torch.Generator().manual_seed(953)));rb=q.T*sv[:3,None]/np.sqrt(2)
 for lr in (.01,.1,.5):
  for key,basis in [('learn',B),('random',rb)]:
   c=fit_code(base,b,basis,sx,sy,lr);_,loss=eval_code(base,b,basis,c,tx,ty)
   if loss<best[key][0]:best[key]=(loss,lr)
  m=fit_full(base,sx,sy,lr)
  with torch.no_grad():z=m(tx);loss=float(nn.functional.cross_entropy(z,ty))
  if loss<best['full'][0]:best['full']=(loss,lr)
 return base_lr,best['learn'][1],best['random'][1],best['full'][1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked',nargs=4,type=float);a=p.parse_args();torch.set_num_threads(1);settings=tune() if a.locked is None else tuple(a.locked);seeds=[53] if a.development else ([223,239,257] if a.fresh else [53,223,239,257]);print(json.dumps({'selected_settings':settings,'runs':[run(s,*settings) for s in seeds]},indent=2))
if __name__=='__main__':main()
