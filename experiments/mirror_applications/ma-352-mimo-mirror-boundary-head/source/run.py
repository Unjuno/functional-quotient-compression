"""MIMO boundary-head compression with a common frozen hidden trunk."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D=16;H=64;K=8;NTR=256;NTE=1024;UPDATES=1000;DEV=[35221,35222];FRESH=[35231,35232,35233]
torch.set_num_threads(1)
def world(seed):
 r=np.random.default_rng(seed);w1=(r.standard_normal((D,H),dtype=np.float32)/np.sqrt(D)).astype(np.float32);b1=(r.standard_normal(H,dtype=np.float32)*.1).astype(np.float32);head=(r.standard_normal(H,dtype=np.float32)*.25).astype(np.float32);ang=r.uniform(0,2*np.pi,K).astype(np.float32)
 teacher=np.repeat(head[None],K,axis=0);c=np.cos(ang);s=np.sin(ang);teacher[:,0]=c*head[0]-s*head[1];teacher[:,1]=s*head[0]+c*head[1]
 def make(n):
  x=r.standard_normal((K,n,D),dtype=np.float32);h=np.maximum(x@w1+b1,0);p=1/(1+np.exp(-np.clip(np.einsum('knh,kh->kn',h,teacher),-30,30)));y=r.binomial(1,p).astype(np.float32);return x,y
 xt,yt=make(NTR);xe,ye=make(NTE);xc=r.standard_normal((1024,D),dtype=np.float32)
 return dict(seed=seed,w1=w1,b1=b1,head=head,angles=ang,teacher=teacher,xt=xt,yt=yt,xe=xe,ye=ye,xc=xc)
def train(a,method):
 xt=torch.tensor(a['xt']);yt=torch.tensor(a['yt']);w1=torch.tensor(a['w1']);b1=torch.tensor(a['b1']);ht=torch.relu(torch.einsum('knd,dh->knh',xt,w1)+b1);torch.manual_seed(a['seed']+{'mimo':5,'mirror':6,'generic':7,'shared':8}[method]);st=time.perf_counter()
 if method=='mimo':p=nn.Parameter(torch.zeros(K,H));params=[p]
 elif method=='mirror':p=nn.Parameter(torch.randn(H)*.05);ang=nn.Parameter(torch.rand(K)*2*torch.pi);params=[p,ang]
 elif method=='generic':p=nn.Parameter(torch.zeros(H));u=nn.Parameter(torch.randn(H)*.025);v=nn.Parameter(torch.randn(H)*.025);coef=nn.Parameter(torch.randn(K,2)*.05);params=[p,u,v,coef]
 else:p=nn.Parameter(torch.zeros(H));params=[p]
 opt=torch.optim.Adam(params,lr=.02,weight_decay=.05)
 for _ in range(UPDATES):
  opt.zero_grad()
  if method=='mimo':w=p
  elif method=='mirror':
   ang= params[1];w=p[None].expand(K,-1);c=ang.cos();s=ang.sin();w=torch.cat([(c*p[0]-s*p[1])[:,None],(s*p[0]+c*p[1])[:,None],p[None,2:].expand(K,-1)],1)
  elif method=='generic':w=p[None]+params[3][:,0,None]*params[1][None]+params[3][:,1,None]*params[2][None]
  else:w=p[None].expand(K,-1)
  loss=nn.functional.binary_cross_entropy_with_logits(torch.einsum('knh,kh->kn',ht,w),yt);loss.backward();opt.step()
 sec=time.perf_counter()-st;return [q.detach().numpy() for q in params],sec,float(loss.detach())
def build(method,state):
 if method=='mimo':return state[0]
 if method=='mirror':
  p,ang=state;c=np.cos(ang);s=np.sin(ang);w=np.repeat(p[None],K,axis=0);w[:,0]=c*p[0]-s*p[1];w[:,1]=s*p[0]+c*p[1];return w
 if method=='generic':p,u,v,co=state;return p[None]+co[:,0,None]*u[None]+co[:,1,None]*v[None]
 return np.repeat(state[0][None],K,axis=0)
def pack(method,a,state):
 arr=[a['w1'],a['b1']]+state;meta={'method':method,'shapes':[list(x.shape) for x in arr],'dtype':'float32','members':K,'version':1};j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();return b'MA352\0'+struct.pack('<I',len(j))+j+b''.join(np.asarray(x,dtype=np.float32).tobytes() for x in arr)
def metrics(method,a,w):
 x=torch.tensor(a['xe']);y=torch.tensor(a['ye']);w1=torch.tensor(a['w1']);b1=torch.tensor(a['b1']);h=torch.relu(torch.einsum('knd,dh->knh',x,w1)+b1);wt=torch.tensor(w);logit=torch.einsum('knh,kh->kn',h,wt);p=torch.sigmoid(logit);nll=float(nn.functional.binary_cross_entropy(p,y));brier=float(((p-y)**2).mean());acc=float(((p>=.5)==y).float().mean());ece=0.
 for k in range(K):
  q=p[k].numpy();t=a['ye'][k]
  for i in range(10):
   lo=i/10;hi=(i+1)/10;mask=(q>=lo)&(q<(hi if i<9 else 1.00001))
   if mask.any():ece+=mask.mean()*abs(float(q[mask].mean())-float(t[mask].mean()))/K
 xc=torch.tensor(a['xc']);hc=torch.relu(xc@w1+b1);lp=(hc@wt.T).numpy().T;cm=np.corrcoef(lp);corr=float(np.nanmean(cm[np.triu_indices(K,1)])) if np.isfinite(cm[np.triu_indices(K,1)]).any() else 1.0;pp=1/(1+np.exp(-np.clip(lp,-30,30)));dis=float(np.mean(np.abs(pp[:,None,:]-pp[None,:,:]))/2)
 return nll,brier,ece,acc,corr,dis
def latency(method,a,state):
 st=time.perf_counter();w=torch.tensor(build(method,state));x=torch.tensor(a['xc']);h=torch.relu(x@torch.tensor(a['w1'])+torch.tensor(a['b1']));_=h@w.T;return time.perf_counter()-st
def run(phase):
 rows=[]
 for seed in DEV if phase=='development' else FRESH:
  a=world(seed)
  for method in ('mimo','mirror','generic','shared'):
   state,sec,loss=train(a,method);w=build(method,state);blob=pack(method,a,state);nll,brier,ece,acc,corr,dis=metrics(method,a,w);lat=latency(method,a,state);headbytes=sum(np.asarray(x).nbytes for x in state)
   rows.append({'phase':phase,'world':seed,'method':method,'serialized_bytes':len(blob),'head_state_bytes':headbytes,'train_examples':K*NTR,'optimizer_updates':UPDATES,'active_compute_proxy':K*NTR*D*H*UPDATES,'train_wall_time_s':sec,'one_batch_inference_wall_time_s':lat,'nll':nll,'brier':brier,'ece_10':ece,'accuracy':acc,'member_logit_correlation':corr,'pairwise_disagreement':dis,'payload_sha256':hashlib.sha256(blob).hexdigest()})
 path=ROOT/f'{phase.upper()}_RESULTS.csv'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
