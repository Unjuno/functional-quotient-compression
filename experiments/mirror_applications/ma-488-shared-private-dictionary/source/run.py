import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[48800,48801];FRESH=[48810,48811,48812];SEEDS=[0,1,2];E=64;D=32;R=4;FRACS=[.25,.5,.75];TH=[.05,.1,.2,.4]
def world(w,s,frac):
 g=torch.Generator().manual_seed(w*100003+s*7919+int(frac*1000));base=torch.randn(D,generator=g)*.2;basis=torch.linalg.qr(torch.randn(D,R,generator=g)).Q;coef=torch.randn(E,R,generator=g)*.2;y=base+coef@basis.T;n=int(E*frac);ids=torch.randperm(E,generator=g)[:n];priv=torch.randn(n,D,generator=g)*.25;y[ids]+=priv
 return base,basis,y

def pack(method,base,basis,y,tau=.1):
 c=(y-base)@basis;shared=base+c@basis.T;res=y-shared
 if method=='dense':o={'method':method,'targets':y}
 elif method=='shared':o={'method':method,'base':base,'basis':basis,'coeff':c}
 elif method=='all_private':o={'method':method,'base':base,'basis':basis,'coeff':c,'residual':res}
 else:
  norms=res.norm(dim=1);ids=norms>tau;o={'method':method,'base':base,'basis':basis,'coeff':c,'private_ids':ids.nonzero().flatten().to(torch.uint8),'private_residual':res[ids]}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()
def decode(o):
 if o['method']=='dense':return o['targets']
 y=o['base']+o['coeff']@o['basis'].T
 if o['method']=='all_private':y+=o['residual']
 if o['method']=='adaptive':y[o['private_ids'].long()]+=o['private_residual']
 return y
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  vals=[]
  for tau in TH:
   errs=[];bytes_=[]
   for w in DEV:
    for s in SEEDS:
     b,ba,y=world(w,s,.5);blob=pack('adaptive',b,ba,y,tau);o=torch.load(io.BytesIO(blob),weights_only=False);errs.append(float((decode(o)-y).norm()/(y.norm()+1e-12)));bytes_.append(len(blob))
   vals.append({'tau':tau,'error':statistics.mean(errs),'bytes':statistics.mean(bytes_)})
  pick=min((x for x in vals if x['error']<=.05),key=lambda x:x['bytes'],default=min(vals,key=lambda x:x['error']));(ART/'selection.json').write_text(json.dumps({'sweep':vals,'tau':pick['tau']},indent=2)+'\n');print(json.dumps(pick));return
 tau=json.loads((ART/'selection.json').read_text())['tau'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   for frac in FRACS:
    base,basis,y=world(w,s,frac)
    for m in ['dense','shared','all_private','adaptive']:
     blob=pack(m,base,basis,y,tau);o=torch.load(io.BytesIO(blob),weights_only=False);err=float((decode(o)-y).norm()/(y.norm()+1e-12));priv=len(o.get('private_ids',torch.arange(E))) if m=='adaptive' else E if m=='all_private' else 0;path=PAY/f'{w}_{s}_{int(frac*100)}_{m}.pt';path.write_bytes(blob);rows.append({'world':w,'seed':s,'heterogeneity':frac,'method':m,'threshold':tau,'normalized_rmse':err,'private_count':priv,'payload_bytes':len(blob),'bytes_per_function':len(blob)/E,'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'tau':tau}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
