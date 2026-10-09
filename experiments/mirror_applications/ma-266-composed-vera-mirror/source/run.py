import csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];D=32;DEV=[26600,26601];FRESH=[26610,26611,26612];SEEDS=[0,1,2];torch.set_num_threads(2);CODES={'identity':1,'scale_sum':2,'sequential_vera':3,'mirror_composed':4,'generic_product_law':5,'explicit_matrix':6};NAMES={v:k for k,v in CODES.items()}
def bank(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+266);u=torch.randn(D,generator=g);u=u/u.norm();a=(torch.rand(2,generator=g)*1.6)-.6;K=torch.outer(u,u);I=torch.eye(D);W1=I+a[0]*K;W2=I+a[1]*K;Wc=W1@W2;ac=a.sum()+a.prod();return u,a,I,K,W1,W2,Wc,ac
def inputs(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+901);return torch.randn(1024,D,generator=g)
def f32bytes(x):return x.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes()
def pack(name,u,a,W):
 code=CODES[name];head=struct.pack('<4sBBBB',b'MA26',1,code,D,0)
 if name=='identity':body=b''
 elif name=='scale_sum':body=f32bytes(u)+f32bytes(a.sum().reshape(1))
 elif name=='sequential_vera':body=f32bytes(u)+f32bytes(a)
 elif name in ('mirror_composed','generic_product_law'):body=f32bytes(u)+f32bytes(a.reshape(1))
 else:body=f32bytes(W)
 return head+body
def decode(blob):
 magic,version,code,d,flags=struct.unpack('<4sBBBB',blob[:8]);assert magic==b'MA26' and version==1 and d==D
 vals=np.frombuffer(blob[8:],dtype='<f4').copy();t=torch.from_numpy(vals);name=NAMES[code]
 if name=='identity':return name,{'W':torch.eye(D)}
 if name=='explicit_matrix':return name,{'W':t.reshape(D,D)}
 if name=='sequential_vera':
  u=t[:D];a=t[D:D+2];K=torch.outer(u,u);I=torch.eye(D);return name,{'W1':I+a[0]*K,'W2':I+a[1]*K}
 u=t[:D];alpha=t[D];K=torch.outer(u,u);return name,{'W':torch.eye(D)+alpha*K}
def nrmse(a,b):return float((a-b).square().mean().sqrt()/b.square().mean().sqrt())
def run(phase):
 rows=[]
 for w in DEV if phase=='development' else FRESH:
  for s in SEEDS:
   u,a,I,K,W1,W2,Wc,ac=bank(w,s);x=inputs(w,s);target=x@Wc;objs={'identity':(None,None),'scale_sum':(u,a.sum().reshape(1)),'sequential_vera':(u,a),'mirror_composed':(u,ac.reshape(1)),'generic_product_law':(u,ac.reshape(1)),'explicit_matrix':(None,Wc)}
   for name,(vec,code) in objs.items():
    aa=code if code is not None else torch.empty(0);blob=pack(name,vec if vec is not None else torch.empty(0),aa,Wc);path=ROOT/'artifacts'/'payloads_binary'/f'{w}_{s}_{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob);td=time.perf_counter();dn,ds=decode(blob);dec=time.perf_counter()-td
    if name=='sequential_vera':fn=lambda x=x,ds=ds:x@ds['W1']@ds['W2']
    else:fn=lambda x=x,ds=ds:x@ds['W']
    pred=fn()
    for _ in range(5):_=fn()
    times=[]
    for _ in range(20):
     ta=time.perf_counter();_=fn();times.append(time.perf_counter()-ta)
    rows.append({'world':w,'seed':s,'method':name,'nrmse':nrmse(pred,target),'payload_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2])),'decode_seconds':dec,'apply_seconds':sorted(times)[10],'apply_steps':2 if name=='sequential_vera' else 1})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
