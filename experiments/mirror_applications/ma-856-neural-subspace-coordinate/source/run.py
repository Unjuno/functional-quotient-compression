"""PA231-style line subspace training on sklearn Digits, CPU only."""
import argparse,io,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
ALPHAS=np.linspace(0.,1.,17,dtype=np.float32); EPOCHS=100; BATCH=64; WIDTH=32; THREADS=1

def splits():
 d=load_digits();x=d.data.astype(np.float32)/16.;y=d.target.astype(np.int64);ids=np.arange(len(y))
 tr,rest=train_test_split(ids,test_size=.4,random_state=420,stratify=y)
 va,te=train_test_split(rest,test_size=.5,random_state=421,stratify=y[rest])
 return (torch.tensor(x[tr]),torch.tensor(y[tr])),(torch.tensor(x[va]),torch.tensor(y[va])),(torch.tensor(x[te]),torch.tensor(y[te]))

class Net(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(64,WIDTH);self.fc2=nn.Linear(WIDTH,10)
 def forward(self,x):return self.fc2(torch.relu(self.fc1(x)))

def cosine_sq(a,b):
 x=torch.cat([p.reshape(-1) for p in a.parameters()]);y=torch.cat([p.reshape(-1) for p in b.parameters()]);return (torch.dot(x,y)/(torch.linalg.norm(x)*torch.linalg.norm(y)+1e-12)).square()

def train_line(seed,lr,beta,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=splits();a,b=Net(),Net();opt=torch.optim.SGD(list(a.parameters())+list(b.parameters()),lr=lr,momentum=.9,weight_decay=1e-4)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
 g=torch.Generator().manual_seed(seed+99);updates=examples=0;started=__import__('time').perf_counter()
 for ep in range(epochs):
  order=torch.randperm(len(x),generator=g)
  for ids in order.split(BATCH):
   if len(ids)<2:continue
   xb,yb=x[ids],y[ids];alpha=torch.rand((),generator=g)
   pa=dict(a.named_parameters());pb=dict(b.named_parameters());mix={k:(1-alpha)*pa[k]+alpha*pb[k] for k in pa}
   logits=functional_call(a,mix,(xb,));loss=nn.functional.cross_entropy(logits,yb)+beta*cosine_sq(a,b)
   opt.zero_grad();loss.backward();opt.step();updates+=1;examples+=len(ids)
  sched.step()
 return a,b,updates,examples,__import__('time').perf_counter()-started

def train_single(seed,lr,epochs=EPOCHS):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);(x,y),_,_=splits();m=Net();opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);g=torch.Generator().manual_seed(seed+99);up=ex=0
 for _ in range(epochs):
  for ids in torch.randperm(len(x),generator=g).split(BATCH):
   xb,yb=x[ids],y[ids];loss=nn.functional.cross_entropy(m(xb),yb);opt.zero_grad();loss.backward();opt.step();up+=1;ex+=len(ids)
  sched.step()
 return m,up,ex

def evaluate_logits(m,x):
 with torch.no_grad():return m(x)
def metrics(z,y):return float((z.argmax(1)==y).float().mean()),float(nn.functional.cross_entropy(z,y))
def line_metrics(a,b,x,y):
 rows=[];probs=[];preds=[];maxdiff=0.
 with torch.no_grad():
  sa,sb=a.state_dict(),b.state_dict()
  for alpha in ALPHAS:
   mcode=2*float(alpha)-1;adecode=(mcode+1.)/2.
   pn={k:(1-float(alpha))*sa[k]+float(alpha)*sb[k] for k in sa}
   pm={k:(1-adecode)*sa[k]+adecode*sb[k] for k in sa}
   zn=functional_call(a,pn,(x,));zm=functional_call(a,pm,(x,));maxdiff=max(maxdiff,float((zn-zm).abs().max()))
   acc,ce=metrics(zn,y);rows.append((float(alpha),acc,ce));probs.append(zn.softmax(-1));preds.append(zn.argmax(-1))
  endpoint_dis=float((preds[0]!=preds[-1]).float().mean())
  pairdis=float(np.mean([float((preds[i]!=preds[j]).float().mean()) for i in range(len(preds)) for j in range(i+1,len(preds))]))
  ensemble=metrics(torch.stack(probs).mean(0).clamp_min(1e-9).log(),y)
  probdist=float(np.mean([float((probs[i]-probs[j]).abs().mean()) for i in range(len(probs)) for j in range(i+1,len(probs))]))
 return rows,endpoint_dis,pairdis,ensemble,probdist,maxdiff

def serial(obj):
 b=io.BytesIO();torch.save(obj,b);return len(b.getvalue()),b.getvalue()
def roundtrip(blob):
 loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);return loaded

def run_seed(seed,lr,beta):
 torch.set_num_threads(THREADS);_,_,(xt,yt)=splits();start=time.perf_counter()
 a,b,up,ex,ttrain=train_line(seed,lr,beta)
 rows,edis,pdis,ens,pdist,mdiff=line_metrics(a,b,xt,yt)
 # PA231 beta=1 default and beta=0 diversity ablation share initialization/data-order seed.
 a1,b1,u1,e1,t1=train_line(seed,lr,1.);rows1,ed1,pd1,ens1,pd1v,diff1=line_metrics(a1,b1,xt,yt)
 a0,b0,u0,e0,t0=train_line(seed,lr,0.);rows0,ed0,pd0,ens0,pd0v,diff0=line_metrics(a0,b0,xt,yt)
 single,su,se=train_single(seed+1000,lr);zs=evaluate_logits(single,xt);single_metric=metrics(zs,yt)
 i1,_,_=train_single(seed+2000,lr);i2,_,_=train_single(seed+3000,lr);zi=[evaluate_logits(m,xt) for m in (i1,i2)]
 indep_acc=float(np.mean([metrics(z,yt)[0] for z in zi]));indep_ce=float(np.mean([metrics(z,yt)[1] for z in zi]));indep_dis=float((zi[0].argmax(1)!=zi[1].argmax(1)).float().mean())
 def pair_payload(x,y,beta_value,coord_key,coords):
  return {'endpoint_a':{k:v.cpu() for k,v in x.state_dict().items()},'endpoint_b':{k:v.cpu() for k,v in y.state_dict().items()},coord_key:torch.tensor(coords,dtype=torch.float32),'architecture':'64-32-10-ReLU','normalization':'digits/16','beta':beta_value}
 payload_obj=pair_payload(a,b,beta,'alpha_codes',ALPHAS);size,blob=serial(payload_obj)
 native=roundtrip(blob);la,lb=Net(),Net();la.load_state_dict(native['endpoint_a']);lb.load_state_dict(native['endpoint_b']);reload_rows,*_=line_metrics(la,lb,xt,yt)
 mirror_obj=pair_payload(a,b,beta,'view_codes',2*ALPHAS-1);mirror_size,mirror_blob=serial(mirror_obj);mirror_loaded=roundtrip(mirror_blob);ma,mb=Net(),Net();ma.load_state_dict(mirror_loaded['endpoint_a']);mb.load_state_dict(mirror_loaded['endpoint_b']);mirror_rows,*_=line_metrics(ma,mb,xt,yt)
 mat_models=[{name:(1-float(alpha))*a.state_dict()[name].cpu()+float(alpha)*b.state_dict()[name].cpu() for name in a.state_dict()} for alpha in ALPHAS]
 mat_size,mat_blob=serial({'models':mat_models,'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 ind_size,_=serial({'models':[i1.state_dict(),i2.state_dict()],'architecture':'64-32-10-ReLU','normalization':'digits/16'})
 mat_loaded=roundtrip(mat_blob)['models'];mat_diff=0.
 with torch.no_grad():
  for alpha,st in zip(ALPHAS,mat_loaded):
   zref=functional_call(a,{k:(1-float(alpha))*a.state_dict()[k]+float(alpha)*b.state_dict()[k] for k in a.state_dict()},(xt,))
   zmat=functional_call(a,st,(xt,));mat_diff=max(mat_diff,float((zref-zmat).abs().max()))
 p=sum(v.numel() for v in a.parameters());decode_macs=len(ALPHAS)*2*p;query_macs=len(ALPHAS)*len(yt)*(64*WIDTH+WIDTH*10)
 useful=sum(acc>=single_metric[0]-.02 for _,acc,_ in rows)
 return dict(seed=seed,lr=lr,beta=beta,alpha_metrics=rows,beta1_metrics=rows1,beta0_metrics=rows0,endpoint_disagreement=edis,pairwise_disagreement=pdis,mean_probability_distance=pdist,ensemble_accuracy=ens[0],ensemble_nll=ens[1],useful_alpha_count=useful,standard_accuracy=single_metric[0],standard_nll=single_metric[1],independent_mean_accuracy=indep_acc,independent_mean_nll=indep_ce,independent_pair_disagreement=indep_dis,native_line_bytes=size,mirror_line_bytes=mirror_size,beta1_line_bytes=serial(pair_payload(a1,b1,1.,'alpha_codes',ALPHAS))[0],beta0_line_bytes=serial(pair_payload(a0,b0,0.,'alpha_codes',ALPHAS))[0],materialized_17_line_bytes=mat_size,independent_2_model_bytes=ind_size,decode_macs=decode_macs,query_macs=query_macs,train_updates=up,train_examples=ex,beta1_train_updates=u1,beta0_train_updates=u0,standard_train_updates=su,independent_train_updates=2*((len(splits()[0][1])+BATCH-1)//BATCH)*EPOCHS,train_wall_time_s=ttrain,beta1_train_wall_time_s=t1,beta0_train_wall_time_s=t0,endpoint_params=p,max_native_mirror_logit_diff=mdiff,max_reload_line_logit_diff=max(abs(x[1]-y[1]) for x,y in zip(rows,reload_rows)),max_materialized_roundtrip_logit_diff=mat_diff,wall_time_s=time.perf_counter()-start)

def tune():
 _,(xv,yv),_=splits();best=None
 for lr in (.01,.03,.1):
  for beta in (.1,1.):
   a,b,_,_,_=train_line(31,lr,beta);rows,*_=line_metrics(a,b,xv,yv);score=np.mean([ce for _,_,ce in rows]);
   if best is None or score<best[0]:best=(score,lr,beta)
 return best[1],best[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--locked-lr',type=float);p.add_argument('--locked-beta',type=float);a=p.parse_args();torch.set_num_threads(THREADS)
 if a.locked_lr is not None or a.locked_beta is not None:
  if a.locked_lr is None or a.locked_beta is None: raise ValueError('provide both locked settings')
  lr,beta=a.locked_lr,a.locked_beta
 else: lr,beta=tune()
 seeds=[31] if a.development else ([61,79,97] if a.fresh else [31,61,79,97]);print(json.dumps({'selected_lr':lr,'selected_beta':beta,'alpha_grid':ALPHAS.tolist(),'runs':[run_seed(s,lr,beta) for s in seeds]},indent=2))
if __name__=='__main__':main()
