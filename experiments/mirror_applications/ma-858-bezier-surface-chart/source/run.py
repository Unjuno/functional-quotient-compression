"""MA-858 quadratic Bezier surface over four rotated Digits domains."""
import argparse,io,json,random,time
import numpy as np,torch
from torch import nn
from torch.func import functional_call
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
DEG=2;WIDTH=32;EPOCHS=60;BATCH=64;GRID=np.linspace(0,1,5,dtype=np.float32)
COORDS=np.array([(u,v) for u in GRID for v in GRID],dtype=np.float32)
def data():
 d=load_digits();imgs=d.images.astype(np.float32)/16.;labels=d.target.astype(np.int64); domains=np.stack([imgs,np.rot90(imgs,1,axes=(1,2)),np.rot90(imgs,2,axes=(1,2)),np.rot90(imgs,3,axes=(1,2))],axis=1).reshape(len(labels),4,64);ids=np.arange(len(labels));tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=labels);va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=labels[rest]);return (torch.tensor(domains[tr]),torch.tensor(labels[tr])),(torch.tensor(domains[va]),torch.tensor(labels[va])),(torch.tensor(domains[te]),torch.tensor(labels[te]))
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))
def bern(u,v):
 bu=torch.tensor([(1-u)**2,2*u*(1-u),u*u],dtype=torch.float32);bv=torch.tensor([(1-v)**2,2*v*(1-v),v*v],dtype=torch.float32);return torch.outer(bu,bv)
def domain_weights(u,v):return np.array([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],np.float64)
def select_batch(x,y,src,u,v,g):
 w=torch.tensor(domain_weights(u,v),dtype=torch.float32);dom=torch.multinomial(w,len(src),replacement=True,generator=g);return x[src,dom],y[src]
def train_surface(seed,lr,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();nets=[Net() for _ in range(9)];opt=torch.optim.SGD([p for n in nets for p in n.parameters()],lr=lr,momentum=.9,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);g=torch.Generator().manual_seed(seed+137);steps=examples=0;start=time.perf_counter()
 for _ in range(epochs):
  order=torch.randperm(len(y),generator=g)
  for src in order.split(BATCH):
   if len(src)<2:continue
   u=float(torch.rand((),generator=g));v=float(torch.rand((),generator=g));xb,yb=select_batch(x,y,src,u,v,g);w=bern(u,v);states=[n.state_dict(keep_vars=True) for n in nets];mix={k:sum(w[i,j]*states[3*i+j][k] for i in range(3) for j in range(3)) for k in states[0]};loss=nn.functional.cross_entropy(functional_call(nets[0],mix,(xb,)),yb);opt.zero_grad();loss.backward();opt.step();steps+=1;examples+=len(src)
  sched.step()
 return nets,steps,examples,time.perf_counter()-start
def train_one(seed,lr,domain=None,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);g=torch.Generator().manual_seed(seed+137);steps=examples=0
 for _ in range(epochs):
  order=torch.randperm(len(y),generator=g)
  for src in order.split(BATCH):
   if len(src)<2:continue
   dom=torch.randint(0,4,(len(src),),generator=g) if domain is None else torch.full((len(src),),domain,dtype=torch.long);loss=nn.functional.cross_entropy(m(x[src,dom]),y[src]);opt.zero_grad();loss.backward();opt.step();steps+=1;examples+=len(src)
  sched.step()
 return m,steps,examples
def sample_eval(x,y,coord,n=360):
 u,v=map(float,coord);n=min(n,len(y));w=domain_weights(u,v);counts=np.floor(w*n).astype(int);counts[np.argsort(w)[-1]]+=n-counts.sum();g=np.random.default_rng(2424);src=g.choice(len(y),n,replace=False);domains=[]
 for k,c in enumerate(counts):domains.extend([k]*int(c))
 g.shuffle(domains);return x[torch.tensor(src),torch.tensor(domains)],y[torch.tensor(src)]
def eval_surface(nets,split):
 x,y=split;states=[n.state_dict() for n in nets];out=[]
 with torch.no_grad():
  for coord in COORDS:
   u,v=map(float,coord);w=bern(u,v);mix={k:sum(w[i,j]*states[3*i+j][k] for i in range(3) for j in range(3)) for k in states[0]};xb,yb=sample_eval(x,y,coord);z=functional_call(nets[0],mix,(xb,));out.append((float((z.argmax(1)==yb).float().mean()),float(nn.functional.cross_entropy(z,yb))))
 return out
def serialize(o):b=io.BytesIO();torch.save(o,b);return len(b.getvalue()),b.getvalue()
def run(seed,lr):
 torch.set_num_threads(1);_,_,test=data();start=time.perf_counter();nets,steps,examples,tw=train_surface(seed,lr);metrics=eval_surface(nets,test);(tx,ty)=test
 pooled,pu,pe=train_one(seed+1000,lr,None);with_pool=None
 with torch.no_grad():pz=pooled(tx.reshape(-1,64));py=ty.repeat_interleave(4);stdacc=float((pz.argmax(1)==py).float().mean());stdnll=float(nn.functional.cross_entropy(pz,py))
 indstart=time.perf_counter();ind=[train_one(seed+2000+i,lr,i)[0] for i in range(4)];indwall=time.perf_counter()-indstart;domacc=[]
 with torch.no_grad():
  for i,m in enumerate(ind):domacc.append(float((m(tx[:,i]).argmax(1)==ty).float().mean()))
 soup=Net();soup.load_state_dict({k:sum(m.state_dict()[k]/4 for m in ind) for k in ind[0].state_dict()});
 with torch.no_grad():sz=soup(tx.reshape(-1,64));soupacc=float((sz.argmax(1)==ty.repeat_interleave(4)).float().mean())
 control={'points':[{k:v.cpu() for k,v in n.state_dict().items()} for n in nets],'coords':COORDS.copy(),'architecture':'64-32-10-ReLU','normalization':'digits/16','surface':'tensor-product quadratic Bernstein 3x3'};size,blob=serialize(control);loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);reloadnets=[Net() for _ in range(9)]
 for n,s in zip(reloadnets,loaded['points']):n.load_state_dict(s)
 reloadmetrics=eval_surface(reloadnets,test);mirdiff=0.;reloadlogit=0.
 with torch.no_grad():
  st=[n.state_dict() for n in nets];xx=tx[:32,0]
  for u,v in COORDS:
   uv=np.array([u,v],np.float32);m=2*uv-1;d=(m+1)/2;w=bern(float(u),float(v));wd=bern(float(d[0]),float(d[1]));pn={k:sum(w[i,j]*st[3*i+j][k] for i in range(3) for j in range(3)) for k in st[0]};pm={k:sum(wd[i,j]*st[3*i+j][k] for i in range(3) for j in range(3)) for k in st[0]};zn=functional_call(nets[0],pn,(xx,));zm=functional_call(nets[0],pm,(xx,));zr=functional_call(reloadnets[0],{k:sum(w[i,j]*reloadnets[3*i+j].state_dict()[k] for i in range(3) for j in range(3)) for k in st[0]},(xx,));mirdiff=max(mirdiff,float((zn-zm).abs().max()));reloadlogit=max(reloadlogit,float((zn-zr).abs().max()))
 mats=[]
 for u,v in COORDS:
  w=bern(float(u),float(v));mats.append({k:sum(w[i,j]*nets[3*i+j].state_dict()[k] for i in range(3) for j in range(3)) for k in nets[0].state_dict()})
 matbytes,matblob=serialize({'models':mats,'architecture':'64-32-10-ReLU','normalization':'digits/16'});matloaded=torch.load(io.BytesIO(matblob),map_location='cpu',weights_only=False)['models'];matlogit=0.
 with torch.no_grad():
  for uv,stmat in zip(COORDS,matloaded):
   ww=bern(float(uv[0]),float(uv[1]));ref={k:sum(ww[i,j]*nets[3*i+j].state_dict()[k] for i in range(3) for j in range(3)) for k in nets[0].state_dict()};zr=functional_call(nets[0],ref,(tx[:32,0],));zm=functional_call(nets[0],stmat,(tx[:32,0],));matlogit=max(matlogit,float((zr-zm).abs().max()))
 indbytes,_=serialize({'models':[m.state_dict() for m in ind],'architecture':'64-32-10-ReLU','normalization':'digits/16'});stdbytes,_=serialize({'state':pooled.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 # Bilinear interpolation of four independent corner checkpoints is a native competing surface.
 indstates=[m.state_dict() for m in ind];bil=[]
 with torch.no_grad():
  for u,v in COORDS:
   dw=domain_weights(float(u),float(v));mix={k:sum(float(dw[i])*indstates[i][k] for i in range(4)) for k in indstates[0]};xb,yb=sample_eval(tx,ty,(u,v));z=functional_call(ind[0],mix,(xb,));bil.append((float((z.argmax(1)==yb).float().mean()),float(nn.functional.cross_entropy(z,yb))))
 return {'seed':seed,'lr':lr,'grid_metrics':metrics,'bilinear_independent_grid_metrics':bil,'standard_accuracy':stdacc,'standard_nll':stdnll,'independent_corner_accuracy':domacc,'soup_accuracy':soupacc,'control_payload_bytes':size,'materialized_25_payload_bytes':matbytes,'independent_4_payload_bytes':indbytes,'standard_payload_bytes':stdbytes,'train_updates':steps,'train_examples':examples,'standard_updates':pu,'standard_examples':pe,'train_wall_time_s':tw,'independent_train_wall_time_s':indwall,'control_blend_macs':9*2410*25,'surface_grid_query_macs':25*360*(64*WIDTH+WIDTH*10),'mirror_max_logit_diff':mirdiff,'reload_max_logit_diff':reloadlogit,'materialized_reload_max_logit_diff':matlogit,'reload_max_accuracy_diff':max(abs(a[0]-b[0]) for a,b in zip(metrics,reloadmetrics)),'wall_time_s':time.perf_counter()-start}
def tune():
 _,dev,_=data();best=None
 for lr in (.03,.1):
  n,*_=train_surface(47,lr);m=eval_surface(n,dev);score=np.mean([c for a,c in m]);
  if best is None or score<best[0]:best=(score,lr)
 return best[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked-lr',type=float);a=p.parse_args();torch.set_num_threads(1);lr=a.locked_lr if a.locked_lr is not None else tune();seeds=[47] if a.development else ([181,199,211] if a.fresh else [47,181,199,211]);print(json.dumps({'selected_lr':lr,'coords':COORDS.tolist(),'runs':[run(s,lr) for s in seeds]},indent=2))
if __name__=='__main__':main()
