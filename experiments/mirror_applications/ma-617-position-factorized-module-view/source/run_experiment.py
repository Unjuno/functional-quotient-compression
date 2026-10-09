#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,platform,time,itertools
from pathlib import Path
import torch
M,D,Q=3,2,256
SEQ=list(itertools.product(range(M),repeat=3));HELD=[s for s in SEQ if (s[0]+2*s[1]+3*s[2])%3==0]
def rot(a):
 c=torch.cos(torch.as_tensor(a));s=torch.sin(torch.as_tensor(a));return torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
def conj(w,a):q=rot(a);return q@w@q.T
def conj_coef(w,cs):
 c,s=cs;q=torch.stack((torch.stack((c,-s)),torch.stack((s,c))));return q@w@q.T
def world(seed):
 g=torch.Generator().manual_seed(seed);bank=torch.randn(M,D,D,generator=g)*.4;phase=(torch.rand(3,generator=g)*2-1)*.8
 weights=[]
 for seq in SEQ:weights.append(torch.stack([conj(bank[m],phase[p]) for p,m in enumerate(seq)]))
 return bank,phase,torch.stack(weights)
def forward(x,w):
 h=x
 for i in range(3):
  h=h@w[i].T
  if i<2:h=torch.relu(h)
 return h
def pack(method,seed,bank,phase,coefs,targets):
 d={'method':method,'seed':seed,'bank':bank.half(),'routes':torch.tensor(SEQ,dtype=torch.uint8),'meta':{'format':'MA617-v1','modules':M,'positions':3,'orders':len(SEQ)}}
 if method=='mirror':d['phase']=phase.half()
 if method=='coeff':d['coeff']=coefs.half()
 if method=='independent':d={'method':method,'seed':seed,'orders':torch.tensor(SEQ,dtype=torch.uint8),'weights':targets.half(),'meta':{'format':'MA617-v1'}}
 b=io.BytesIO();torch.save(d,b);return b.getvalue()
def decode(method,bank,phase,coefs):
 if method=='independent':return torch.stack([torch.stack([conj(bank[m],phase[p]) for p,m in enumerate(seq)]) for seq in SEQ])
 # Decode the phase address or the direct native coefficient pair.
 if method=='mirror':return torch.stack([torch.stack([conj(bank[m],phase[p]) for p,m in enumerate(seq)]) for seq in SEQ])
 return torch.stack([torch.stack([conj_coef(bank[m],coefs[p]) for p,m in enumerate(seq)]) for seq in SEQ])
def run(seed,phase_name):
 bank,phase,targets=world(seed);coefs=torch.stack((torch.cos(phase),torch.sin(phase)),-1);x=torch.randn(Q,D,generator=torch.Generator().manual_seed(seed+3));root=Path(__file__).resolve().parents[1]/f'runs/{phase_name}_payloads';root.mkdir(parents=True,exist_ok=True);records=[]
 for method in ['path','mirror','coeff','independent']:
  if method=='path':w=torch.stack([torch.stack([bank[m] for m in seq]) for seq in SEQ])
  elif method=='independent':w=targets
  else:w=decode(method,bank,phase,coefs)
  ys=torch.stack([forward(x,z) for z in targets]);yp=torch.stack([forward(x,z) for z in w]);err=((yp-ys)**2).mean((1,2))/(ys.var((1,2),unbiased=False)+1e-12);blob=pack(method,seed,bank,phase,coefs,targets);fp=root/f'{seed}_{method}.pt';fp.write_bytes(blob)
  lat=[]
  for _ in range(5):
   st=time.perf_counter()
   if method in ('mirror','coeff'):
    # decode all position-module realizations prior to the shared ordered executor
    mats=torch.stack([torch.stack([conj(bank[m],phase[p]) if method=='mirror' else conj_coef(bank[m],coefs[p]) for m in range(M)]) for p in range(3)])
    z=torch.stack([torch.stack([mats[p,m] for p,m in enumerate(seq)]) for seq in SEQ])
   elif method=='path':z=torch.stack([torch.stack([bank[m] for m in seq]) for seq in SEQ])
   else:z=targets
   for ww in z:forward(x,ww)
   lat.append(time.perf_counter()-st)
  r={'method':method,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'file':str(fp.relative_to(root.parent)),'all_mean_nrmse2':float(err.mean()),'heldout_mean_nrmse2':float(err[[SEQ.index(s) for s in HELD]].mean()),'heldout_max_nrmse2':float(err[[SEQ.index(s) for s in HELD]].max()),'mac_proxy_per_order':29.333333333333332 if method in ('mirror','coeff') else 24,'wall_time_s':float(torch.tensor(lat).median()),'optimizer_updates':0,'examples':len(SEQ)*Q}
  records.append(r)
 return {'seed':seed,'phase':phase_name,'orders':len(SEQ),'heldout_orders':len(HELD),'records':records}
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','fresh'],required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1);seeds=[61701,61702] if a.phase=='dev' else [61711,61712,61713];ws=[run(s,a.phase) for s in seeds];d={'experiment_id':'MA-617','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','worlds':ws};o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(d,indent=2)+'\n')
 with (o.parent.parent/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.writer(f);wr.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for w in ws:
   for r in w['records']:wr.writerow([a.phase,w['seed'],r['method'],r['bytes'],r['examples'],0,r['mac_proxy_per_order'],r['wall_time_s'],'heldout_mean_nrmse2',r['heldout_mean_nrmse2'],'heldout_max_nrmse2',r['heldout_max_nrmse2'],r['sha256']])
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()
