#!/usr/bin/env python3
"""MA-602 sparse private collision repair and Mirror View controls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,sklearn,torch,torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
INPUT,HIDDEN,OUTPUT=64,128,10
UPDATES,BATCH,LR=800,128,.01
METHODS=('dense','hash_2048','hash_4096','givens_only','collision_residual','collision_residual_givens','random_residual_givens','rank1_residual')
DIGEST='faf2d98f61250c1cdc0b114323133d3c58da04f5c5d28bb78e4bde3c936a75b1'

def data():
 d=load_digits();raw=np.ascontiguousarray(d.data.astype(np.float32));y=np.ascontiguousarray(d.target.astype(np.int64));return raw/16.,y,hashlib.sha256(raw.tobytes()+y.tobytes()).hexdigest()
def hash_map(bucket_count,seed):
 r=np.random.default_rng(int(seed)&0xffffffff);idx=r.integers(0,bucket_count,(INPUT,HIDDEN),dtype=np.int64);sg=r.choice(np.asarray([-1.,1.],np.float32),(INPUT,HIDDEN));return idx,sg
def collision_positions(indices):
 buckets={}
 for flat,b in enumerate(indices.reshape(-1)):buckets.setdefault(int(b),[]).append(flat)
 return np.asarray([pos for b in sorted(buckets) for pos in buckets[b][1:]],dtype=np.int64)
def exception_indices(method,idx,seed):
 if method in ('collision_residual','collision_residual_givens'):return collision_positions(idx)
 if method=='random_residual_givens':
  k=len(collision_positions(idx));return np.sort(np.random.default_rng(seed+4103).choice(INPUT*HIDDEN,size=k,replace=False)).astype(np.int64)
 return np.empty(0,dtype=np.int64)
def givens(x,angles):
 a,b=x[...,0::2],x[...,1::2];c,s=torch.cos(angles),torch.sin(angles);z=torch.empty_like(x);z[...,0::2]=c*a-s*b;z[...,1::2]=s*a+c*b;return z
def init(method,bucket_count,seed,exception_count):
 torch.manual_seed(seed);p={}
 if method=='dense':p['w1']=torch.nn.Parameter(torch.empty(INPUT,HIDDEN));torch.nn.init.kaiming_uniform_(p['w1'].T,a=0.,nonlinearity='relu')
 else:p['theta']=torch.nn.Parameter(torch.randn(bucket_count)*(2./INPUT)**.5)
 p['b1']=torch.nn.Parameter(torch.zeros(HIDDEN));p['w2']=torch.nn.Parameter(torch.empty(HIDDEN,OUTPUT));p['b2']=torch.nn.Parameter(torch.zeros(OUTPUT));torch.nn.init.xavier_uniform_(p['w2'].T)
 if method in ('givens_only','collision_residual_givens','random_residual_givens'):p['angles']=torch.nn.Parameter(torch.zeros(INPUT//2))
 if method in ('collision_residual','collision_residual_givens','random_residual_givens'):p['exceptions']=torch.nn.Parameter(torch.zeros(exception_count))
 if method=='rank1_residual':p['u']=torch.nn.Parameter(torch.randn(INPUT,1)*.01);p['v']=torch.nn.Parameter(torch.randn(1,HIDDEN)*.01)
 return p
def current_w1(p,m,idx,sg,exidx):
 if m=='dense':return p['w1']
 w=(p['theta'][torch.as_tensor(idx)]*torch.as_tensor(sg)).reshape(-1)
 if m in ('collision_residual','collision_residual_givens','random_residual_givens'):w=w.index_add(0,torch.as_tensor(exidx),p['exceptions'])
 w=w.reshape(INPUT,HIDDEN)
 if m=='rank1_residual':w=w+p['u']@p['v']
 return w
def logits(p,m,x,idx,sg,exidx):
 w=current_w1(p,m,idx,sg,exidx)
 if m in ('givens_only','collision_residual_givens','random_residual_givens'):x=givens(x,p['angles'])
 h=F.relu(x@w+p['b1']);return h@p['w2']+p['b2']
def train(m,b,x,y,seed,mi,hseed):
 idx,sg=(None,None) if m=='dense' else hash_map(b,hseed);exidx=exception_indices(m,idx,hseed) if idx is not None else np.empty(0,np.int64);p=init(m,b,seed*100+mi,len(exidx));opt=torch.optim.AdamW(list(p.values()),lr=LR,weight_decay=0.);rng=np.random.default_rng(seed*1000);xt=torch.tensor(x);yt=torch.tensor(y);trace=[];st=time.perf_counter()
 for i in range(UPDATES):
  ids=rng.integers(0,len(xt),BATCH);loss=F.cross_entropy(logits(p,m,xt[ids],idx,sg,exidx),yt[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if i in (0,UPDATES-1):trace.append(float(loss.detach()))
 return p,idx,sg,exidx,time.perf_counter()-st,trace
def state(p,m,b,seed,hseed,exidx):
 a={k:v.detach().cpu().numpy().astype(np.float16) for k,v in p.items()};a.update({'method':np.frombuffer(m.encode(),np.uint8),'bucket_count':np.asarray([b if m!='dense' else 0],np.int32),'hash_seed':np.asarray([hseed if m!='dense' else 0],np.uint32),'exception_indices':np.asarray(exidx,dtype=np.uint16),'run_seed':np.asarray([seed],np.uint32),'shape':np.asarray([INPUT,HIDDEN,OUTPUT],np.int32),'schema':np.asarray([602,1],np.int32)});return a
def load(path):
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 m=bytes(a.pop('method').tolist()).decode();b=int(a.pop('bucket_count')[0]);hs=int(a.pop('hash_seed')[0]);seed=int(a.pop('run_seed')[0]);ex=a.pop('exception_indices').astype(np.int64);a.pop('shape');a.pop('schema');p={k:torch.tensor(v,dtype=torch.float32) for k,v in a.items()};idx,sg=(None,None) if m=='dense' else hash_map(b,hs);return m,b,hs,seed,p,idx,sg,ex
def evaluate(path,x,y):
 m,b,hs,seed,p,idx,sg,ex=load(path);xt=torch.tensor(x);yt=torch.tensor(y);st=time.perf_counter();losses=[];correct=0
 with torch.inference_mode():
  for i in range(0,len(xt),BATCH):
   z=logits(p,m,xt[i:i+BATCH],idx,sg,ex);losses.append(float(F.cross_entropy(z,yt[i:i+BATCH],reduction='sum')));correct+=int((z.argmax(1)==yt[i:i+BATCH]).sum())
 sec=time.perf_counter()-st;return {'accuracy':correct/len(yt),'cross_entropy':sum(losses)/len(yt),'inference_seconds':sec,'examples_per_second':len(yt)/sec}
def run(seed,split,out):
 torch.set_num_threads(1);x,y,dig=data();assert dig==DIGEST;xtr,xte,ytr,yte=train_test_split(x,y,test_size=.25,random_state=seed,stratify=y);hs=(seed*7919+120)&0xffffffff;out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];allst=time.perf_counter()
 for mi,m in enumerate(METHODS):
  b=4096 if m=='hash_4096' else 2048;p,idx,sg,ex,train_s,trace=train(m,b,xtr,ytr,seed,mi,hs);path=out/(m+'.npz');np.savez(path,**state(p,m,b,seed,hs,ex));ev=evaluate(path,xte,yte);entries=0 if m=='dense' else INPUT*HIDDEN;viewops=(INPUT//2)*6 if m in ('givens_only','collision_residual_givens','random_residual_givens') else (INPUT+HIDDEN if m=='rank1_residual' else 0)
  rows.append({'method':m,'bucket_count':None if m=='dense' else b,'exception_count':int(len(ex)),'exception_index_bytes':int(len(ex)*2),'exception_value_bytes':int(len(ex)*2),'serialized_bytes':path.stat().st_size,'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'optimizer_updates':UPDATES,'examples_seen':UPDATES*BATCH,'initial_and_final_train_loss':trace,'training_seconds':train_s,'base_macs_per_example':INPUT*HIDDEN+HIDDEN*OUTPUT,'hash_expansion_lookups_per_model_load':entries,'private_exception_adds_per_example':int(len(ex)),'extra_view_ops_per_example':viewops,**ev})
 report={'experiment_id':'MA-602','seed':seed,'split':split,'dataset':'sklearn.load_digits','sklearn_version':sklearn.__version__,'dataset_tensor_sha256':dig,'split_random_state':seed,'hash_seed':hs,'methods':rows,'common_compute':{'device':'cpu','threads':1,'updates_per_method':UPDATES,'input_examples_seen_per_method':UPDATES*BATCH,'total_wall_seconds':time.perf_counter()-allst}};(out/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--split',choices=['dev','fresh'],required=True);a.add_argument('--out',type=Path,required=True);z=a.parse_args();run(z.seed,z.split,z.out)
if __name__=='__main__':main()
