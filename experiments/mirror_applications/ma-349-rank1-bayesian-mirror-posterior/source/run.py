"""Small variational posterior screen: angular Mirror coordinate vs rank-1 weight noise."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];DEV=[34921,34922];FRESH=[34931,34932,34933];UPDATES=1500;MC_TRAIN=16;MC_TEST=64
torch.set_num_threads(1)
def world(seed):
 r=np.random.default_rng(seed);angle=float(r.uniform(.2,.8));wtrue=np.array([1.2*np.cos(angle),1.2*np.sin(angle)],np.float32)
 def data(n,ood=False):
  x=r.standard_normal((n,2),dtype=np.float32)
  if ood:x[:,0]+=1.;x[:,1]*=1.5
  p=1/(1+np.exp(-x@wtrue));y=r.binomial(1,p).astype(np.float32)
  return x,y
 xt,yt=data(64);xi,yi=data(1024);xo,yo=data(1024,True)
 return dict(seed=seed,angle=angle,wtrue=wtrue,xt=xt,yt=yt,xi=xi,yi=yi,xo=xo,yo=yo)
def logit_loss(logits,y):return torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
def fit_map(a):
 x=torch.tensor(a['xt']);y=torch.tensor(a['yt']);torch.manual_seed(a['seed']+100);w=nn.Parameter(torch.zeros(2));b=nn.Parameter(torch.zeros(()));opt=torch.optim.Adam([w,b],lr=.04);st=time.perf_counter()
 for _ in range(UPDATES):opt.zero_grad();loss=logit_loss(x@w+b,y);loss.backward();opt.step()
 return [w.detach().numpy(),np.array([b.detach().item()],np.float32)],time.perf_counter()-st

def fit_mirror(a):
 x=torch.tensor(a['xt']);y=torch.tensor(a['yt']);torch.manual_seed(a['seed']+200);mu=nn.Parameter(torch.tensor(.5));ls=nn.Parameter(torch.tensor(-.4));rr=nn.Parameter(torch.tensor(.5));b=nn.Parameter(torch.zeros(()));opt=torch.optim.Adam([mu,ls,rr,b],lr=.025);st=time.perf_counter()
 for _ in range(UPDATES):
  eps=torch.randn(MC_TRAIN);std=ls.clamp(-4,1).exp();ang=mu+std*eps;rad=nn.functional.softplus(rr)+.05;w=rad*torch.stack([ang.cos(),ang.sin()],-1);logits=x@w.T+b;loss=logit_loss(logits,y[:,None].expand_as(logits)).mean();kl=.5*(mu.square()+std.square()-1-2*std.log())+.5*((rad-1.2)/.5).square();total=loss+kl/len(y);opt.zero_grad();total.backward();opt.step()
 return [np.array([nn.functional.softplus(rr).detach().item()+.05,mu.detach().item(),ls.detach().item(),b.detach().item()],np.float32)],time.perf_counter()-st

def fit_rank1(a):
 x=torch.tensor(a['xt']);y=torch.tensor(a['yt']);torch.manual_seed(a['seed']+300);center=nn.Parameter(torch.zeros(2));psi=nn.Parameter(torch.tensor(.5));zm=nn.Parameter(torch.tensor(0.));ls=nn.Parameter(torch.tensor(-.3));b=nn.Parameter(torch.zeros(()));opt=torch.optim.Adam([center,psi,zm,ls,b],lr=.025);st=time.perf_counter()
 for _ in range(UPDATES):
  eps=torch.randn(MC_TRAIN);std=ls.clamp(-4,1).exp();z=zm+std*eps;direction=torch.stack([psi.cos(),psi.sin()]);w=center[None]+z[:,None]*direction[None];logits=x@w.T+b;loss=logit_loss(logits,y[:,None].expand_as(logits)).mean();kl=.5*(zm.square()+std.square()-1-2*std.log())+.5*center.square().sum();total=loss+kl/len(y);opt.zero_grad();total.backward();opt.step()
 return [np.concatenate([center.detach().numpy(),np.array([psi.detach().item(),zm.detach().item(),ls.detach().item(),b.detach().item()],np.float32)])],time.perf_counter()-st

def samples(method,state,n=MC_TEST):
 if method=='map':return np.repeat(state[0][None,:],n,axis=0),np.repeat(state[1],n)
 if method=='mirror':
  rad,mu,ls,b=state[0];g=np.random.default_rng(7701);ang=g.normal(mu,np.exp(np.clip(ls,-4,1)),n);return rad*np.stack([np.cos(ang),np.sin(ang)],-1),np.full(n,b,np.float32)
 c0,c1,psi,zm,ls,b=state[0];g=np.random.default_rng(7702);z=g.normal(zm,np.exp(np.clip(ls,-4,1)),n);d=np.array([np.cos(psi),np.sin(psi)]);center=np.array([c0,c1]);return center[None,:]+z[:,None]*d[None,:],np.full(n,b,np.float32)
def metrics(method,state,a,split):
 x=a['xi'] if split=='iid' else a['xo'];y=a['yi'] if split=='iid' else a['yo'];w,b=samples(method,state);t=time.perf_counter();logits=x@w.T+b[None,:];probs=(1/(1+np.exp(-np.clip(logits,-30,30)))).mean(1);lat=time.perf_counter()-t
 eps=1e-12;nll=float(-np.mean(y*np.log(probs+eps)+(1-y)*np.log(1-probs+eps)));brier=float(np.mean((probs-y)**2));acc=float(np.mean((probs>=.5)==y));ece=0.
 for lo in np.linspace(0,1,11)[:-1]:
  mask=(probs>=lo)&(probs<(lo+.1 if lo<.9 else 1.000001))
  if mask.any():ece+=mask.mean()*abs(float(probs[mask].mean())-float(y[mask].mean()))
 return nll,brier,ece,acc,lat

def pack(method,state):
 meta={'method':method,'shapes':[list(a.shape) for a in state],'dtype':'float32','posterior_samples':MC_TEST,'version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA349\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in state)
def run(phase):
 rows=[]
 for seed in DEV if phase=='development' else FRESH:
  a=world(seed);models={'map':(*fit_map(a),),'mirror':(*fit_mirror(a),),'rank1_bnn':(*fit_rank1(a),)}
  for name,(state,trainsec) in models.items():
   blob=pack(name,state)
   for split in ('iid','ood'):
    nll,brier,ece,acc,lat=metrics(name,state,a,split)
    rows.append({'phase':phase,'world':seed,'method':name,'split':split,'serialized_bytes':len(blob),'train_examples':64,'optimizer_updates':UPDATES if name!='map' else UPDATES,'mc_samples':MC_TEST,'active_compute_proxy':64*2*MC_TEST,'train_wall_time_s':trainsec,'inference_wall_time_s':lat,'nll':nll,'brier':brier,'ece_10':ece,'accuracy':acc,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
