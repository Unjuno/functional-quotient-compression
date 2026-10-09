#!/usr/bin/env python3
"""MA-601 distillation screen for hashed logical attention heads."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
H,D,HD,T,V=4,32,8,8,32
BUCKETS,UPDATES,BATCH=128,1500,32
METHODS=('independent_heads','gqa2','mqa','tied_hash','salted_hash','mirror_givens_hash','diagonal_hash','rank1_hash')

def make_world(seed):
 r=np.random.default_rng(seed+7001);x=r.normal(0,1,(640,T,D)).astype(np.float32)
 q=r.normal(0,.2,(H,D,HD)).astype(np.float32);k=r.normal(0,.2,(H,D,HD)).astype(np.float32);v=r.normal(0,.2,(H,D,HD)).astype(np.float32);wo=r.normal(0,.6,(H*HD,V)).astype(np.float32)
 xt=torch.tensor(x);qt=torch.tensor(q);kt=torch.tensor(k);vt=torch.tensor(v);wt=torch.tensor(wo)
 with torch.inference_mode():logits,ctx,att=attend(xt,qt,kt,vt,wt);target=logits.softmax(-1).numpy()
 return x[:512],target[:512],x[512:],target[512:],q,k,v,wo

def attend(x,q,k,v,wo):
 qq=torch.einsum('btd,hdf->bhtf',x,q);kk=torch.einsum('btd,hdf->bhtf',x,k);vv=torch.einsum('btd,hdf->bhtf',x,v)
 a=torch.softmax(qq@kk.transpose(-1,-2)/(HD**.5),-1);ctx=a@vv;flat=ctx.transpose(1,2).reshape(len(x),T,H*HD);return flat@wo,ctx,a

def hmap(seed):
 r=np.random.default_rng(int(seed)&0xffffffff);return r.integers(0,BUCKETS,(D,HD),dtype=np.int64),r.choice(np.array([-1.,1.],np.float32),(D,HD))
def ptrain(method,seed,teacher):
 torch.manual_seed(seed*101+17);q,k,v,wo=teacher;p={'wo':torch.tensor(wo)}
 if method=='independent_heads':
  for n,w in [('q',q),('k',k),('v',v)]:p[n]=torch.nn.Parameter(torch.randn_like(torch.tensor(w))*.1)
 elif method=='gqa2':
  p['q']=torch.nn.Parameter(torch.randn(H,D,HD)*.1);p['k']=torch.nn.Parameter(torch.randn(2,D,HD)*.1);p['v']=torch.nn.Parameter(torch.randn(2,D,HD)*.1)
 elif method=='mqa':
  p['q']=torch.nn.Parameter(torch.randn(H,D,HD)*.1);p['k']=torch.nn.Parameter(torch.randn(1,D,HD)*.1);p['v']=torch.nn.Parameter(torch.randn(1,D,HD)*.1)
 elif method in ('tied_hash','salted_hash','mirror_givens_hash','diagonal_hash','rank1_hash'):
  p['theta']=torch.nn.Parameter(torch.randn(3,BUCKETS)*.05)
  if method=='mirror_givens_hash':p['angles']=torch.nn.Parameter(torch.zeros(H,D//2))
  elif method=='diagonal_hash':p['scale']=torch.nn.Parameter(torch.ones(H,D))
  elif method=='rank1_hash':
   p['ru']=torch.nn.Parameter(torch.randn(3,H,D,1)*.01);p['rv']=torch.nn.Parameter(torch.randn(3,H,1,HD)*.01)
 else:raise ValueError(method)
 return p

def maps(method,seed):
 if method=='salted_hash':return [[hmap(seed+1+h*11+r*101) for r in range(3)] for h in range(H)]
 return [[hmap(seed+1+r*101) for r in range(3)] for h in range(H)]

def current(p,method,mp):
 if method in ('independent_heads','gqa2','mqa'):
  q=p['q'];k=p['k'];v=p['v']
  if method=='gqa2':k=k.repeat_interleave(2,dim=0);v=v.repeat_interleave(2,dim=0)
  elif method=='mqa':k=k.expand(H,-1,-1);v=v.expand(H,-1,-1)
  return q,k,v
 ws=[]
 for r in range(3):
  heads=[]
  for h in range(H):
   ix,sg=mp[h][r];w=p['theta'][r][torch.as_tensor(ix)]*torch.as_tensor(sg)
   if method=='rank1_hash':w=w+p['ru'][r,h]@p['rv'][r,h]
   heads.append(w)
  ws.append(torch.stack(heads))
 return ws[0],ws[1],ws[2]

def forward(x,p,method,mp):
 q,k,v=current(p,method,mp)
 if method in ('mirror_givens_hash','diagonal_hash'):
  qs=[];ks=[];vs=[]
  for h in range(H):
   xh=givens(x,p['angles'][h]) if method=='mirror_givens_hash' else x*p['scale'][h]
   qs.append(xh@q[h]);ks.append(xh@k[h]);vs.append(xh@v[h])
  qq,kk,vv=torch.stack(qs,1),torch.stack(ks,1),torch.stack(vs,1)
 else:
  qq=torch.einsum('btd,hdf->bhtf',x,q);kk=torch.einsum('btd,hdf->bhtf',x,k);vv=torch.einsum('btd,hdf->bhtf',x,v)
 a=torch.softmax(qq@kk.transpose(-1,-2)/(HD**.5),-1);ctx=a@vv;flat=ctx.transpose(1,2).reshape(len(x),T,H*HD);return flat@p['wo'],ctx,a

def givens(x,a):
 e,o=x[...,0::2],x[...,1::2];c,s=torch.cos(a),torch.sin(a);z=torch.empty_like(x);z[...,0::2]=c*e-s*o;z[...,1::2]=s*e+c*o;return z

def dump(path,p,method,seed,mp):
 arr={k:v.detach().cpu().numpy().astype(np.float16) for k,v in p.items()};map_seeds=np.asarray([[[seed+1+h*11+r*101 if method=='salted_hash' else seed+1+r*101 for r in range(3)] for h in range(H)]],np.uint32) if method in ('tied_hash','salted_hash','mirror_givens_hash','diagonal_hash','rank1_hash') else np.empty((0,),np.uint32);arr.update({'method':np.frombuffer(method.encode(),np.uint8),'seed':np.asarray([seed],np.uint32),'maps_seed':map_seeds,'shape':np.asarray([H,D,HD,T,V],np.int32),'schema':np.asarray([601,1],np.int32)});path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arr);return path.stat().st_size
def restore(path):
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 method=bytes(a.pop('method').tolist()).decode();seed=int(a.pop('seed')[0]);ms=a.pop('maps_seed');a.pop('shape');a.pop('schema');p={k:torch.tensor(v,dtype=torch.float32) for k,v in a.items()};mp=[[hmap(int(ms[0][h][r])) for r in range(3)] for h in range(H)] if len(ms) else [[hmap(0) for _ in range(3)] for _ in range(H)];return method,seed,p,mp

def diversity(ctx,att):
 def pairs(z):
  vals=[]
  for i in range(H):
   for j in range(i+1,H):vals.append(float(F.cosine_similarity(z[:,i].reshape(1,-1),z[:,j].reshape(1,-1)).item()))
  return float(np.mean(vals))
 return pairs(ctx),pairs(att)
def metrics(path,x,y):
 m,s,p,mp=restore(path);xt=torch.tensor(x);yt=torch.tensor(y);start=time.perf_counter()
 with torch.inference_mode():logits,ctx,att=forward(xt,p,m,mp);ce=float(-(yt*F.log_softmax(logits,-1)).sum(-1).mean());agree=float((logits.argmax(-1)==yt.argmax(-1)).float().mean());hc,ac=diversity(ctx,att)
 sec=time.perf_counter()-start;return {'teacher_cross_entropy':ce,'top1_agreement':agree,'head_context_cosine_mean':hc,'attention_map_cosine_mean':ac,'inference_seconds':sec,'token_positions_per_second':len(x)*T/sec}
def run(seed,split,out):
 torch.set_num_threads(1);xtr,ytr,xte,yte,q,k,v,wo=make_world(seed);teacher=(q,k,v,wo);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];allstart=time.perf_counter();teacher_log=float(-(torch.tensor(yte)*torch.tensor(np.log(np.maximum(yte,1e-30)))).sum(-1).mean())
 for mi,m in enumerate(METHODS):
  p=ptrain(m,seed+mi,teacher);mp=maps(m,seed*7919+600);opt=torch.optim.AdamW([v for v in p.values() if isinstance(v,torch.nn.Parameter)],lr=.003,weight_decay=0.);xr=torch.tensor(xtr);yr=torch.tensor(ytr);rng=np.random.default_rng(seed*1000);st=time.perf_counter();loss0=0.
  for step in range(UPDATES):
   ids=rng.integers(0,len(xr),BATCH);logits,_,_=forward(xr[ids],p,m,mp);loss=-(yr[ids]*F.log_softmax(logits,-1)).sum(-1).mean()
   if step==0:loss0=float(loss.detach())
   opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  trainsec=time.perf_counter()-st;path=out/(m+'.npz');nbytes=dump(path,p,m,seed*7919+600,mp);ev=metrics(path,xte,yte);qmac=H*D*HD*3;extra=H*(D//2)*6 if m=='mirror_givens_hash' else (H*D if m=='diagonal_hash' else H*3*(D+HD) if m=='rank1_hash' else 0);look=H*3*D*HD if m in ('independent_heads','gqa2','mqa') else 3*D*HD*(H if m=='salted_hash' else 1)
  kvheads=2 if m=='gqa2' else (1 if m=='mqa' else H);qmac=(H+2*kvheads)*D*HD
  rows.append({'method':m,'serialized_bytes':nbytes,'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_seconds':trainsec,'initial_train_ce':loss0,'updates':UPDATES,'sequences_seen':UPDATES*BATCH,'qkv_mac_proxy':qmac,'extra_view_ops_per_token':extra,'hash_expansion_lookups':look,**ev})
 report={'experiment_id':'MA-601','world_seed':seed,'split':split,'shape':[H,D,HD,T,V],'teacher_entropy':teacher_log,'methods':rows,'common_compute':{'device':'cpu','threads':1,'updates_per_method':UPDATES,'wall_seconds':time.perf_counter()-allstart}};(out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--split',choices=['dev','fresh'],required=True);a.add_argument('--out',type=Path,required=True);z=a.parse_args();run(z.seed,z.split,z.out)
if __name__=='__main__':main()
