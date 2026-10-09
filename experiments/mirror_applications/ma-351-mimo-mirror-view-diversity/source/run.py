"""Linear multi-input/output MIMO analogue with trained shared-view alternatives."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];K=8;D=8;NTR=256;NTE=1024;UPDATES=1000;DEV=[35121,35122];FRESH=[35131,35132,35133]
torch.set_num_threads(1)
def world(seed):
 r=np.random.default_rng(seed);base=(r.standard_normal(D,dtype=np.float32)*.55).astype(np.float32);angles=r.uniform(0,2*np.pi,K).astype(np.float32)
 weights=np.repeat(base[None,:],K,axis=0);c=np.cos(angles);s=np.sin(angles);weights[:,0]=c*base[0]-s*base[1];weights[:,1]=s*base[0]+c*base[1]
 def make(n):
  x=r.standard_normal((K,n,D),dtype=np.float32);p=1/(1+np.exp(-np.clip(np.einsum('knd,kd->kn',x,weights),-30,30)));y=r.binomial(1,p).astype(np.float32);return x,y
 xt,yt=make(NTR);xe,ye=make(NTE);xc=r.standard_normal((1024,D),dtype=np.float32)
 return dict(seed=seed,teacher=weights,angles=angles,xt=xt,yt=yt,xe=xe,ye=ye,xc=xc)
def train(a,method):
 xt=torch.tensor(a['xt']);yt=torch.tensor(a['yt']);torch.manual_seed(a['seed']+{'mimo':10,'mirror':20,'generic':30,'shared':40}[method]);st=time.perf_counter()
 if method=='mimo':
  p=nn.Parameter(torch.zeros(K,D));params=[p];
 elif method=='mirror':
  p=nn.Parameter(torch.randn(D)*.1);ang=nn.Parameter(torch.rand(K)*2*torch.pi);params=[p,ang]
 elif method=='generic':
  p=nn.Parameter(torch.zeros(D));u=nn.Parameter(torch.randn(D)*.05);v=nn.Parameter(torch.randn(D)*.05);coef=nn.Parameter(torch.randn(K,2)*.05);params=[p,u,v,coef]
 else:p=nn.Parameter(torch.zeros(D));params=[p]
 opt=torch.optim.Adam(params,lr=.025)
 for _ in range(UPDATES):
  opt.zero_grad()
  if method=='mimo':w=p
  elif method=='mirror':
   w=p[None].repeat(K,1);c=ang.cos();s=ang.sin();w=torch.cat([(c*p[0]-s*p[1])[:,None],(s*p[0]+c*p[1])[:,None],p[None,2:].expand(K,-1)],dim=1)
  elif method=='generic':w=p[None]+coef[:,0,None]*u[None]+coef[:,1,None]*v[None]
  else:w=p[None].expand(K,-1)
  logits=torch.einsum('knd,kd->kn',xt,w);loss=nn.functional.binary_cross_entropy_with_logits(logits,yt);loss.backward();opt.step()
 sec=time.perf_counter()-st
 return [q.detach().numpy() for q in params],sec,float(loss.detach())
def build(method,state):
 if method=='mimo':return state[0]
 if method=='mirror':
  p,ang=state;c=np.cos(ang);s=np.sin(ang);w=np.repeat(p[None],K,axis=0);w[:,0]=c*p[0]-s*p[1];w[:,1]=s*p[0]+c*p[1];return w
 if method=='generic':p,u,v,coef=state;return p[None]+coef[:,0,None]*u[None]+coef[:,1,None]*v[None]
 return np.repeat(state[0][None],K,axis=0)
def pack(method,state):
 meta={'method':method,'shapes':[list(x.shape) for x in state],'dtype':'float32','members':K,'version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA351\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in state)
def metrics(weights,a):
 x=torch.tensor(a['xe']);y=torch.tensor(a['ye']);w=torch.tensor(weights);logit=torch.einsum('knd,kd->kn',x,w);prob=torch.sigmoid(logit);lat=0.0
 nll=float(nn.functional.binary_cross_entropy(prob, y).item());brier=float(((prob-y)**2).mean());acc=float(((prob>=.5)==y).float().mean());ece=0.
 for k in range(K):
  q=prob[k].numpy();t=a['ye'][k]
  for i in range(10):
   lo=i/10;hi=(i+1)/10;mask=(q>=lo)&(q<(hi if i<9 else 1.00001))
   if mask.any():ece+=mask.mean()*abs(float(q[mask].mean())-float(t[mask].mean()))/K
 common=torch.tensor(a['xc']);lp=torch.einsum('nd,kd->kn',common,w).numpy();corr=np.corrcoef(lp);upper=np.triu_indices(K,1);member_corr=float(np.mean(corr[upper]));p=1/(1+np.exp(-np.clip(lp,-30,30)));dis=float(np.mean(np.abs(p[:,None,:]-p[None,:,:]))/2)
 return nll,brier,ece,acc,member_corr,dis,lat
def batch_latency(method,state,a):
 st=time.perf_counter();w=torch.tensor(build(method,state));x=torch.tensor(a['xc']);_ = torch.einsum('nd,kd->kn',x,w);return time.perf_counter()-st
def run(phase):
 rows=[]
 for seed in DEV if phase=='development' else FRESH:
  a=world(seed)
  for method in ('mimo','mirror','generic','shared'):
   state,trainsec,trainloss=train(a,method);w=build(method,state);blob=pack(method,state);vals=metrics(w,a);nll,brier,ece,acc,corr,dis,_=vals;lat=batch_latency(method,state,a)
   rows.append({'phase':phase,'world':seed,'method':method,'serialized_bytes':len(blob),'train_examples':K*NTR,'optimizer_updates':UPDATES,'active_compute_proxy':K*NTE*D+(K if method=='mirror' else 2*K if method=='generic' else 0),'train_wall_time_s':trainsec,'inference_one_batch_wall_time_s':lat,'nll':nll,'brier':brier,'ece_10':ece,'accuracy':acc,'member_logit_correlation':corr,'pairwise_disagreement':dis,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
