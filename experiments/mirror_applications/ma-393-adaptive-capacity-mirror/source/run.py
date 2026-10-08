import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
V,COMMON,RARE,DIM,CLASSES=128,32,96,16,8
TRAIN,VAL,TEST,BATCH,UPDATES=32768,16384,16384,128,256
METHODS=('full16','adaptive4','adaptive5','mirror_angle','scalar_gate')
torch.set_num_threads(1)
def make_data(seed):
 rng=np.random.default_rng(seed+43000);latent=rng.normal(0,.6,(RARE,4)).astype(np.float32);angles=rng.uniform(-1.1,1.1,RARE).astype(np.float32);projection=rng.normal(0,.5,(4,DIM)).astype(np.float32);common=rng.normal(0,.6,(COMMON,DIM)).astype(np.float32);head=rng.normal(0,.7,(CLASSES,DIM)).astype(np.float32)
 rare=np.zeros((RARE,DIM),np.float32)
 for i,z in enumerate(latent):
  c,s=np.cos(angles[i]),np.sin(angles[i]);rot=np.r_[c*z[0]-s*z[1],s*z[0]+c*z[1],z[2:]];rare[i]=rot@projection
 emb=np.concatenate([common,rare]);labels=(emb@head.T).argmax(1).astype(np.int64)
 # common tokens occur 8x as often as rare tokens in training.
 weights=np.r_[np.full(COMMON,8.),np.ones(RARE)];weights/=weights.sum()
 trng=np.random.default_rng(seed+17);train=trng.choice(V,TRAIN,p=weights)
 def balanced(salt,n):
  rg=np.random.default_rng(seed+salt);ids=np.tile(np.arange(V),int(np.ceil(n/V)))[:n];rg.shuffle(ids);return ids
 out={'teacher_angles':angles.tolist(),'train':train}
 for name,ids in [('validation',balanced(29,VAL)),('test',balanced(41,TEST))]:out[name]={'tokens':torch.tensor(ids,dtype=torch.long),'labels':torch.tensor(labels[ids],dtype=torch.long)}
 out['train']={'tokens':torch.tensor(train,dtype=torch.long),'labels':torch.tensor(labels[train],dtype=torch.long)}
 return out
class Model(torch.nn.Module):
 def __init__(self,method,seed):
  super().__init__();self.method=method;torch.manual_seed(seed+METHODS.index(method)*83)
  if method=='full16':self.table=torch.nn.Parameter(torch.randn(V,DIM)*.03)
  else:
   self.common=torch.nn.Parameter(torch.randn(COMMON,DIM)*.03)
   width=5 if method=='adaptive5' else 4
   self.rare=torch.nn.Parameter(torch.randn(RARE,width)*.03);self.proj=torch.nn.Parameter(torch.randn(width,DIM)*.03)
   if method=='mirror_angle':self.angle=torch.nn.Parameter(torch.zeros(RARE))
   if method=='scalar_gate':self.gate=torch.nn.Parameter(torch.ones(RARE))
  self.head=torch.nn.Parameter(torch.randn(CLASSES,DIM)*.03);self.bias=torch.nn.Parameter(torch.zeros(CLASSES))
 def embed(self,t):
  if self.method=='full16':return self.table[t]
  common=t<COMMON;idx=(t-COMMON).clamp_min(0);out=torch.empty((len(t),DIM),device=t.device,dtype=self.common.dtype)
  if common.any():out[common]=self.common[t[common]]
  if (~common).any():
   z=self.rare[idx[~common]]
   if self.method=='mirror_angle':
    a=self.angle[idx[~common]];c,s=torch.cos(a)[:,None],torch.sin(a)[:,None];z=torch.cat([c*z[:,:1]-s*z[:,1:2],s*z[:,:1]+c*z[:,1:2],z[:,2:]],1)
   elif self.method=='scalar_gate':z=z*self.gate[idx[~common],None]
   out[~common]=z@self.proj
  return out
 def forward(self,t):return self.embed(t)@self.head.T+self.bias
def evaluate(model,split):
 with torch.no_grad():
  t=split['tokens'];y=split['labels'];log=model(t);ok=log.argmax(1).eq(y);common=t<COMMON;rare=~common
  e=model.embed(torch.arange(V)).cpu().numpy().astype('<f2')
  return {'accuracy':float(ok.float().mean()),'nll':float(F.cross_entropy(log,y)),'common_accuracy':float(ok[common].float().mean()),'rare_accuracy':float(ok[rare].float().mean()),'common_nll':float(F.cross_entropy(log[common],y[common])),'rare_nll':float(F.cross_entropy(log[rare],y[rare])),'unique_embeddings_fp16':int(len(np.unique(e,axis=0)))}
def state_arrays(m):
 a={'meta':np.asarray([V,COMMON,RARE,DIM,CLASSES,METHODS.index(m.method)],dtype=np.uint16)}
 for k,v in m.state_dict().items():a[k]=v.detach().cpu().numpy().astype('<f2')
 return a
def pack(a,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
  for n in sorted(a):
   b=io.BytesIO();np.lib.format.write_array(b,np.ascontiguousarray(a[n]),allow_pickle=False);i=zipfile.ZipInfo(n+'.npy',date_time=(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_STORED;i.external_attr=0o600<<16;z.writestr(i,b.getvalue())
 raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()
def load(method,seed,a):
 m=Model(method,seed);m.load_state_dict({k:torch.tensor(a[k],dtype=v.dtype) for k,v in m.state_dict().items()});m.eval();return m
def train(method,seed,data,outdir):
 m=Model(method,seed);opt=torch.optim.Adam(m.parameters(),lr=.003);t=data['train']['tokens'];y=data['train']['labels'];rng=np.random.default_rng(seed*31+METHODS.index(method));perm=rng.permutation(len(t));start=time.perf_counter()
 for step in range(UPDATES):
  ix=torch.from_numpy(perm[step*BATCH:(step+1)*BATCH]);log=m(t[ix]);loss=F.cross_entropy(log,y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 wall=time.perf_counter()-start;a=state_arrays(m);path=Path(outdir)/'payloads'/f'development_{seed}_{method}.npz';size,digest=pack(a,path);loaded=load(method,seed,a)
 return {'method':method,'serialized_bytes':size,'payload_sha256':digest,'optimizer_updates':UPDATES,'training_examples_seen':UPDATES*BATCH,'train_wall_s':wall,'validation':evaluate(loaded,data['validation']),'test':evaluate(loaded,data['test'])}
def run(seed,condition,outdir,dest):
 data=make_data(seed);rows=[train(m,seed,data,outdir) for m in METHODS];r={'seed':seed,'condition':condition,'teacher_angles':data['teacher_angles'],'summaries':rows};Path(dest).parent.mkdir(parents=True,exist_ok=True);Path(dest).write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=['development','fresh'],required=True);p.add_argument('--outdir',required=True);p.add_argument('--json',required=True);a=p.parse_args();run(a.seed,a.condition,a.outdir,a.json)
