#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,platform,time
from pathlib import Path
import torch
T,R,D,ALIGN,NQ=32,4,2,24,256

def rot(a):
 c=torch.cos(torch.as_tensor(a));s=torch.sin(torch.as_tensor(a));return torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
def conj(w,a):
 q=rot(a);return q@w@q.T
def world(seed):
 g=torch.Generator().manual_seed(seed);bank=torch.randn(2,R,D,D,generator=g)*.45;routes=torch.randint(R,(T,2),generator=g);phase=(torch.rand(T,2,generator=g)*2-1)*1.1;ws=[]
 for t in range(T):
  if t<ALIGN: z=torch.stack((conj(bank[0,routes[t,0]],phase[t,0]),conj(bank[1,routes[t,1]],phase[t,1])))
  else:z=torch.randn(2,D,D,generator=g)*.45
  ws.append(z)
 return bank,routes,phase,torch.stack(ws)
def forward(x,w):return torch.relu(x@w[0].T)@w[1].T
def decode(bank,routes,phase,coefs,private,method):
 priv={int(i):private[k] for k,i in enumerate(torch.arange(ALIGN,T))} if private.shape[0] else {};out=[]
 for t in range(T):
  if t in priv:out.append(priv[t]);continue
  mats=[]
  for j in range(2):
   w=bank[j,routes[t,j]]
   if method=='mirror':w=conj(w,phase[t,j])
   elif method=='coeff':
    c,s=coefs[t,j];q=torch.stack((torch.stack((c,-s)),torch.stack((s,c))));w=q@w@q.T
   mats.append(w)
  out.append(torch.stack(mats))
 return torch.stack(out)
def payload(method,seed,bank,routes,phase,coefs,priv,targets):
 d={'method':method,'seed':seed,'bank':bank.half(),'routes':routes.to(torch.uint8),'meta':{'format':'MA616-v1','tasks':T,'aligned':ALIGN,'stages':2,'modules_per_stage':R}}
 if method=='mirror':d['phase']=phase[:ALIGN].half()
 if method=='coeff':d['coeff']=coefs[:ALIGN].half()
 if method in ('mirror','coeff'):d['private_indices']=torch.arange(ALIGN,T).to(torch.uint8);d['private']=priv.half()
 if method=='independent':d={'method':method,'seed':seed,'weights':targets.half(),'meta':{'format':'MA616-v1','tasks':T}}
 b=io.BytesIO();torch.save(d,b);return b.getvalue()
def evalone(method,seed,bank,routes,phase,coefs,targets,x,root):
 priv=targets[ALIGN:]
 if method=='independent':pw=targets
 elif method=='path':pw=decode(bank,routes,phase,coefs,torch.empty(0,2,D,D),'path')
 elif method=='mirror':pw=decode(bank,routes,phase,coefs,priv,'mirror')
 else:pw=decode(bank,routes,phase,coefs,priv,'coeff')
 y=torch.stack([forward(x,w) for w in targets]);yp=torch.stack([forward(x,w) for w in pw]);err=((yp-y)**2).mean((1,2))/(y.var((1,2),unbiased=False)+1e-12)
 blob=payload(method,seed,bank,routes,phase,coefs,priv,targets);path=root/f'{seed}_{method}.pt';path.write_bytes(blob)
 lat=[]
 for _ in range(5):
  ts=time.perf_counter()
  if method=='independent':cur=targets
  elif method=='path':cur=decode(bank,routes,phase,coefs,torch.empty(0,2,D,D),'path')
  elif method=='mirror':cur=decode(bank,routes,phase,coefs,priv,'mirror')
  else:cur=decode(bank,routes,phase,coefs,priv,'coeff')
  for w in cur:forward(x,w)
  lat.append(time.perf_counter()-ts)
 return {'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(path.relative_to(root.parent)),'mean_nrmse2':float(err.mean()),'aligned_mean_nrmse2':float(err[:ALIGN].mean()),'aligned_max_nrmse2':float(err[:ALIGN].max()),'unrelated_mean_nrmse2':float(err[ALIGN:].mean()),'mac_proxy_per_query':8 if method in ('path','independent') else 8+32/NQ,'wall_time_s':float(torch.tensor(lat).median()),'optimizer_updates':0,'examples':T*NQ}
def run(seed,phase):
 bank,r,p,w=world(seed);x=torch.randn(NQ,D,generator=torch.Generator().manual_seed(seed+17));c=torch.stack((torch.cos(p),torch.sin(p)),-1);root=Path(__file__).resolve().parents[1]/f'runs/{phase}_payloads';root.mkdir(parents=True,exist_ok=True)
 return {'seed':seed,'phase':phase,'tasks':T,'aligned':ALIGN,'unrelated':T-ALIGN,'records':[evalone(m,seed,bank,r,p,c,w,x,root) for m in ['path','mirror','coeff','independent']]}
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','fresh'],required=True);a.add_argument('--out',required=True);z=a.parse_args();torch.set_num_threads(1);seeds=[61601,61602] if z.phase=='dev' else [61611,61612,61613];ws=[run(s,z.phase) for s in seeds];d={'experiment_id':'MA-616','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':ws};o=Path(z.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n')
 with (o.parent.parent/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.writer(f);wr.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for w in ws:
   for r in w['records']:wr.writerow([z.phase,w['seed'],r['method'],r['bytes'],r['examples'],0,r['mac_proxy_per_query'],r['wall_time_s'],'mean_nrmse2',r['mean_nrmse2'],'aligned_max_nrmse2',r['aligned_max_nrmse2'],r['sha256']])
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
