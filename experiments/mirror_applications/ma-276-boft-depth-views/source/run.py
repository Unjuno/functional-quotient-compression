"""MA-276: isolate depth-step views on a shared recurrent linear block."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];D,K=16,4;DEV,FRESH=[27600,27601],[27610,27611,27612];SEEDS=[0,1,2];STEPS=500
torch.set_num_threads(2)

def rot(a):
 q=torch.eye(D);c,s=torch.cos(a),torch.sin(a);q[0,0]=c;q[1,1]=c;q[0,1]=-s;q[1,0]=s;return q
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+276);x=torch.randn(1024,D,generator=g);v=torch.randn(512,D,generator=g)
 base=torch.eye(D)*.35+torch.randn(D,D,generator=g)*.02
 angles=torch.tensor([-.6,-.2,.2,.6]);targets=[]
 for a in angles:targets.append(base@rot(a))
 # independent stratum: task matrices from unrelated full operators
 independent=torch.stack([torch.randn(D,D,generator=g)*.12 for _ in range(K)])
 return x,v,base,angles,torch.stack(targets),independent
def recur(x,Ws):
 h=x
 for w in Ws:h=h+h@w
 return h
def pack(method,base,codes):
 flat=torch.cat([base.flatten(),codes.flatten()]).float().numpy().tobytes();meta=json.dumps({'method':method,'base':[D,D],'codes':list(codes.shape),'steps':K},sort_keys=True,separators=(',',':')).encode();return b'MA276\0'+struct.pack('<I',len(meta))+meta+flat
def fit(method,x,base,teachers,seed):
 torch.manual_seed(seed);pars=[]
 if method=='tied':p={'base':nn.Parameter(base.clone())}
 elif method=='mirror':p={'base':nn.Parameter(base.clone()),'a':nn.Parameter(torch.zeros(K))}
 elif method=='rank1':p={'base':nn.Parameter(base.clone()),'u':nn.Parameter(torch.zeros(K,D)),'v':nn.Parameter(torch.randn(D)*.01)}
 elif method=='lora':p={'base':nn.Parameter(base.clone()),'A':nn.Parameter(torch.randn(K,D,2)*.01),'B':nn.Parameter(torch.zeros(K,2,D))}
 else:p={'W':nn.Parameter(teachers.clone())}
 opt=torch.optim.Adam(list(p.values()),lr=.02);t=time.perf_counter()
 for _ in range(STEPS):
  opt.zero_grad();ws=[]
  for i in range(K):
   if method=='tied':w=p['base']
   elif method=='mirror':w=p['base']@rot(p['a'][i])
   elif method=='rank1':w=p['base']+torch.outer(p['u'][i],p['v'])
   elif method=='lora':w=p['base']+p['A'][i]@p['B'][i]
   else:w=p['W'][i]
   ws.append(w)
  loss=sum((x@ws[i]-x@teachers[i]).square().mean() for i in range(K));loss.backward();opt.step()
 return {k:v.detach() for k,v in p.items()},time.perf_counter()-t
def decode(method,p):
 if method=='tied':return [p['base']]*K
 if method=='mirror':return [p['base']@rot(p['a'][i]) for i in range(K)]
 if method=='rank1':return [p['base']+torch.outer(p['u'][i],p['v']) for i in range(K)]
 if method=='lora':return [p['base']+p['A'][i]@p['B'][i] for i in range(K)]
 return [p['W'][i] for i in range(K)]
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   x,xv,base,angles,aligned,ind=make(world,seed)
   for stratum,teachers in [('aligned_rotations',aligned),('independent',ind)]:
    for method in ['tied','rank1','lora','mirror','untied']:
     p,sec=fit(method,x,base,teachers,world*31+seed);Ws=decode(method,p)
     # Deterministic inference payload charges shared base and method-specific codes/deltas.
     parts=[v.flatten() for k,v in p.items() if k!='base'];codes=torch.cat(parts) if parts else torch.empty(0)
     blob=pack(method,p.get('base',torch.empty(0)),codes);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{stratum}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
     for step in range(K):
      pred=xv@Ws[step];truth=xv@teachers[step];err=float(torch.linalg.vector_norm(pred-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-12));rows.append({'phase':phase,'world':world,'seed':seed,'stratum':stratum,'method':method,'step':step,'nrmse':err,'payload_bytes':len(blob),'active_macs':len(xv)*D*D,'train_seconds':sec,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
