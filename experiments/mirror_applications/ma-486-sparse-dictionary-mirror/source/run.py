import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[48600,48601];FRESH=[48610,48611,48612];SEEDS=[0,1,2];D=32;M=64;N=128;TOPKS=[2,4,8,16]
g=torch.Generator().manual_seed(486001);DICT=torch.nn.functional.normalize(torch.randn(D,M,generator=g),dim=0)
def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);coef=torch.zeros(N,M);support=[]
 for i in range(N):
  ids=torch.randperm(M,generator=g)[:4];vals=torch.randn(4,generator=g);coef[i,ids]=vals;support.append(ids)
 return coef@DICT.T,coef

def omp(y,k):
 residual=y.clone();ids=[];vals=[]
 for _ in range(k):
  scores=residual@DICT;idx=scores.abs().argmax(1);val=scores.gather(1,idx[:,None]).squeeze(1);ids.append(idx);vals.append(val);residual-=val[:,None]*DICT[:,idx].T
 return torch.stack(ids,1).to(torch.uint8),torch.stack(vals,1)
def payload(method,y,k):
 if method=='dense':o={'method':method,'targets':y.clone()}
 elif method=='sparse':
  ids,vals=omp(y,k);o={'method':method,'dictionary':DICT.clone(),'ids':ids,'values':vals,'N':N,'K':k}
 elif method=='dense_coeff':
  c=torch.linalg.lstsq(DICT,y.T).solution.T;o={'method':method,'dictionary':DICT.clone(),'coeff':c.half()}
 else:
  ids,vals=omp(y,k);vals=torch.sign(vals).to(torch.int8);o={'method':'signed_sparse','dictionary':DICT.clone(),'ids':ids,'signs':vals,'N':N,'K':k}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()
def decode(o):
 if o['method']=='dense':return o['targets']
 if o['method']=='dense_coeff':return o['coeff'].float()@o['dictionary'].T
 z=torch.zeros(o['N'],D)
 for j in range(o['K']):
  idx=o['ids'][:,j].long();v=o['values'][:,j] if o['method']=='sparse' else o['signs'][:,j].float();z+=v[:,None]*o['dictionary'][:,idx].T
 return z
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  summary=[]
  for k in TOPKS:
   es=[]
   for w in DEV:
    for s in SEEDS:
     y,_=world(w,s);b=payload('sparse',y,k);o=torch.load(io.BytesIO(b),weights_only=False);es.append(float((decode(o)-y).norm()/(y.norm()+1e-12)))
   summary.append({'topk':k,'mean_nrmse':statistics.mean(es)})
  (ART/'development_selection.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return
 rows=[]
 for w in FRESH:
  for s in SEEDS:
   y,_=world(w,s)
   for k in TOPKS:
    for method in ['dense','dense_coeff','signed_sparse','sparse']:
     b=payload(method,y,k);o=torch.load(io.BytesIO(b),weights_only=False);err=float((decode(o)-y).norm()/(y.norm()+1e-12));p=PAY/f'{w}_{s}_{method}_K{k}.pt';p.write_bytes(b);rows.append({'world':w,'seed':s,'method':method,'topk':k,'normalized_rmse':err,'payload_bytes':len(b),'bytes_per_function':len(b)/N,'decode_MAC_per_function':k*D*2 if method in ('sparse','signed_sparse') else D*M,'hash':hashlib.sha256(b).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
