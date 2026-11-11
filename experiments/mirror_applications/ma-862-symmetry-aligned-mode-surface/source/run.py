"""Symmetry alignment and bilinear interpolation on rotated Digits models."""
import argparse,io,json,random,time
import numpy as np,torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
WIDTH=32;EPOCHS=60;BATCH=64;GRID=np.linspace(0,1,5,dtype=np.float32);COORDS=np.array([(u,v) for u in GRID for v in GRID],dtype=np.float32)
def data():
 d=load_digits();im=d.images.astype(np.float32)/16.;y=d.target.astype(np.int64);x=np.stack([im,np.rot90(im,1,axes=(1,2)),np.rot90(im,2,axes=(1,2)),np.rot90(im,3,axes=(1,2))],axis=1).reshape(len(y),4,64);ids=np.arange(len(y));tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=y);va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=y[rest]);return (torch.tensor(x[tr]),torch.tensor(y[tr])),(torch.tensor(x[va]),torch.tensor(y[va])),(torch.tensor(x[te]),torch.tensor(y[te]))
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))
def train_domain(seed,lr,dom,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+dom*199);updates=examples=0;st=time.perf_counter()
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();updates+=1;examples+=len(ids)
  sch.step()
 return m,updates,examples,time.perf_counter()-st
def train_pooled(seed,lr,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=data();m=Net();o=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);g=torch.Generator().manual_seed(seed+331);updates=examples=0
 for _ in range(epochs):
  for ids in torch.randperm(len(y),generator=g).split(BATCH):
   if len(ids)<2:continue
   dom=torch.randint(0,4,(len(ids),),generator=g);loss=nn.functional.cross_entropy(m(x[ids,dom]),y[ids]);o.zero_grad();loss.backward();o.step();updates+=1;examples+=len(ids)
  sch.step()
 return m,updates,examples
def activations(m,x):
 with torch.no_grad():return torch.relu(m.fc1(x))
def align_to_reference(ref,model,probe):
 a=activations(ref,probe);b=activations(model,probe);a=a-a.mean(0,keepdim=True);b=b-b.mean(0,keepdim=True);corr=a.T@b/(a.norm(dim=0)[:,None]*b.norm(dim=0)[None,:]+1e-12);rows,cols=linear_sum_assignment(-corr.numpy());perm=torch.tensor(cols[np.argsort(rows)],dtype=torch.long);before=model(probe[:64]).detach().clone()
 with torch.no_grad():model.fc1.weight.copy_(model.fc1.weight[perm]);model.fc1.bias.copy_(model.fc1.bias[perm]);model.fc2.weight.copy_(model.fc2.weight[:,perm])
 after=model(probe[:64]).detach();return perm,float((before-after).abs().max()),float(corr.max(dim=1).values.mean())
def weights(models,coord,aligned):
 u,v=map(float,coord);w=torch.tensor([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v]);states=[m.state_dict() for m in models];return {k:sum(w[i]*states[i][k] for i in range(4)) for k in states[0]}
def batch_for_coord(split,coord):
 x,y=split;u,v=map(float,coord);w=np.array([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v]);n=len(y);counts=np.floor(w*n).astype(int);counts[np.argmax(w)]+=n-counts.sum();rng=np.random.default_rng(812);src=rng.choice(n,n,replace=False);domains=np.concatenate([np.full(c,i,dtype=int) for i,c in enumerate(counts)]);rng.shuffle(domains);return x[torch.tensor(src),torch.tensor(domains)],y[torch.tensor(src)]
def eval_surface(models,split):
 x,y=split;out=[]
 with torch.no_grad():
  for uv in COORDS:
   st=weights(models,uv,True);xb,yb=batch_for_coord(split,uv);z=torch.func.functional_call(models[0],st,(xb,));out.append((float((z.argmax(1)==yb).float().mean()),float(nn.functional.cross_entropy(z,yb))))
 return out
def serialize(o):b=io.BytesIO();torch.save(o,b);return len(b.getvalue()),b.getvalue()
def run(seed,lr):
 torch.set_num_threads(1);(tr,ty),val,test=data();st=time.perf_counter();models=[];up=ex=0;trainwall=0.
 for i in range(4):m,u,e,t=train_domain(seed+13*i,lr,i);models.append(m);up+=u;ex+=e;trainwall+=t
 probe=tr.reshape(-1,64);aligned=[models[0]];perms=[torch.arange(WIDTH)];maxpres=0.;matchscore=[];alignstart=time.perf_counter()
 for m in models[1:]:
  mc=Net();mc.load_state_dict(m.state_dict());p,pres,score=align_to_reference(models[0],mc,probe);aligned.append(mc);perms.append(p);maxpres=max(maxpres,pres);matchscore.append(score)
 alignwall=time.perf_counter()-alignstart;rawmetrics=eval_surface(models,test);almetrics=eval_surface(aligned,test);pool,pu,pe=train_pooled(seed+4000,lr);
 with torch.no_grad():
  px=test[0].reshape(-1,64);py=test[1].repeat_interleave(4);pz=pool(px);poolacc=float((pz.argmax(1)==py).float().mean());poolnll=float(nn.functional.cross_entropy(pz,py))
 corner=[]
 with torch.no_grad():
  for i,m in enumerate(models):corner.append(float((m(test[0][:,i]).argmax(1)==test[1]).float().mean()))
 # Model soups on equally weighted domain mixture.
 def soup_acc(ms):
  sm=Net();sm.load_state_dict({k:sum(m.state_dict()[k]/4 for m in ms) for k in ms[0].state_dict()});xx=test[0].reshape(-1,64);yy=test[1].repeat_interleave(4)
  with torch.no_grad():return float((sm(xx).argmax(1)==yy).float().mean())
 raw_soup=soup_acc(models);aligned_soup=soup_acc(aligned)
 # Pay for permutation indices to decode aligned states from raw checkpoints.
 rawbytes,_=serialize({'models':[m.state_dict() for m in models],'architecture':'64-32-10-ReLU','normalization':'digits/16'});permbytes,permblob=serialize({'permutations':[p for p in perms],'alignment_method':'train activation correlation + Hungarian','reference':0});alignpayload,_=serialize({'raw_models':[m.state_dict() for m in models],'permutations':[p for p in perms],'architecture':'64-32-10-ReLU','normalization':'digits/16'});alignedbytes,_=serialize({'models':[m.state_dict() for m in aligned],'architecture':'64-32-10-ReLU','normalization':'digits/16'});poolbytes,_=serialize({'state':pool.state_dict(),'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 mats=[weights(aligned,uv,True) for uv in COORDS];matbytes,matblob=serialize({'models':mats,'architecture':'64-32-10-ReLU','normalization':'digits/16'});loaded=torch.load(io.BytesIO(matblob),map_location='cpu',weights_only=False)['models'];matdiff=0.
 with torch.no_grad():
  for uv,ms in zip(COORDS,loaded):xb=test[0][:32,0];ref=weights(aligned,uv,True);z=torch.func.functional_call(aligned[0],ref,(xb,));zm=torch.func.functional_call(aligned[0],ms,(xb,));matdiff=max(matdiff,float((z-zm).abs().max()))
 mirrordiff=0.
 with torch.no_grad():
  for u,v in COORDS:
   uv=np.array([u,v],np.float32);m=2*uv-1;d=(m+1)/2;a=weights(aligned,uv,True);b=weights(aligned,d,True);x=test[0][:32,0];mirrordiff=max(mirrordiff,float((torch.func.functional_call(aligned[0],a,(x,))-torch.func.functional_call(aligned[0],b,(x,))).abs().max()))
 return {'seed':seed,'lr':lr,'unaligned_grid_metrics':rawmetrics,'aligned_grid_metrics':almetrics,'endpoint_accuracy':corner,'pooled_accuracy':poolacc,'pooled_nll':poolnll,'pooled_payload_bytes':poolbytes,'pooled_updates':pu,'pooled_examples':pe,'mean_match_correlation':matchscore,'max_alignment_output_preservation_diff':maxpres,'unaligned_soup_accuracy':raw_soup,'aligned_soup_accuracy':aligned_soup,'raw_4model_bytes':rawbytes,'permutation_metadata_bytes':permbytes,'raw_plus_permutation_bytes':alignpayload,'aligned_4model_bytes':alignedbytes,'materialized_25_aligned_views_bytes':matbytes,'training_updates':up,'training_examples':ex,'train_wall_time_s':trainwall,'alignment_wall_time_s':alignwall,'alignment_mac_proxy':4*len(probe)*WIDTH*WIDTH,'blend_mac_proxy':4*2410*25,'grid_query_macs':25*len(test[1])*(64*WIDTH+WIDTH*10),'max_mirror_native_logit_difference':mirrordiff,'max_materialized_reload_logit_difference':matdiff,'total_wall_time_s':time.perf_counter()-st}
def tune():
 _,val,_=data();best=None
 for lr in (.03,.1):
  ms=[train_domain(67+13*i,lr,i)[0] for i in range(4)];nlls=[]
  with torch.no_grad():
   for i,m in enumerate(ms):nlls.append(float(nn.functional.cross_entropy(m(val[0][:,i]),val[1])))
  score=sum(nlls)/4
  if best is None or score<best[0]:best=(score,lr)
 return best[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--locked-lr',type=float);a=p.parse_args();torch.set_num_threads(1);lr=a.locked_lr if a.locked_lr is not None else tune();seeds=[67] if a.development else ([271,283,307] if a.fresh else [67,271,283,307]);print(json.dumps({'selected_lr':lr,'coordinates':COORDS.tolist(),'runs':[run(s,lr) for s in seeds]},indent=2))
if __name__=='__main__':main()
