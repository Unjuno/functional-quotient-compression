import csv,hashlib,io,json,math,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[49200,49201];FRESH=[49220,49221,49222];SEEDS=[0,1,2];C=256;M=16;VOC=8;KS=[2,4,8,16]
def data(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);perm=torch.randperm(M,generator=g);idx=torch.arange(M);modes=torch.stack([idx%VOC,(idx//VOC+idx%VOC)%VOC],1);modes=modes[perm];ctx=[]
 for i in range(C):
  a=(i*5+s)%M;b=(a+1+(i%3))%M;ctx.append([a,b])
 targets=torch.tensor(ctx,dtype=torch.long);return modes,targets

def payload(method,modes,targets,k=16):
 if method=='plan_vq':
  codes=modes[:k].clone();ctx_codes=[]
  for a,b in targets.tolist():ctx_codes.append([a if a<k else -1,b if b<k else -1])
  obj={'method':method,'codebook':codes,'context_codes':torch.tensor(ctx_codes,dtype=torch.int16),'N':C,'K':k}
 elif method=='continuous':
  # Full-precision context-specific probabilities over all joint packet modes.
  probs=torch.zeros(C,M);probs.scatter_(1,targets,.5)
  obj={'method':method,'mode_table':modes,'context_probs':probs,'N':C}
 elif method=='independent':
  p1=torch.zeros(C,VOC);p2=torch.zeros(C,VOC);pairs=modes[targets]
  p1.scatter_add_(1,pairs[:,:,0],torch.full((C,2),.5))
  p2.scatter_add_(1,pairs[:,:,1],torch.full((C,2),.5))
  p1/=p1.sum(1,keepdim=True);p2/=p2.sum(1,keepdim=True)
  obj={'method':method,'p1':p1,'p2':p2,'N':C}
 else:
  obj={'method':'greedy','mode_table':modes,'choice':targets[:,0].to(torch.uint8),'N':C}
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def metrics(o,modes,targets):
 if o['method']=='plan_vq':
  supported=set(range(o['K']));cov=len(supported)/M;valid=1.0;missing=sum(1 for a,b in targets.tolist() if a not in supported and b not in supported)/C;nll=0.
  for a,b in targets.tolist():
   available=int(a in supported)+int(b in supported)
   for target in (a,b):
    prob=(1/available) if target in supported and available else 1e-9
    nll-=.5*math.log(prob)
  return cov,missing,valid,math.log(max(1,len(supported))),nll/C
 if o['method']=='continuous':return 1.,0.,1.,math.log(M),math.log(2)
 if o['method']=='independent':
  # Sampling independent marginals for correlated diagonal token modes; valid pair probability is sum_i p1_i*p2_i.
  valid=0.;covered=set()
  for j,(a,b) in enumerate(modes.tolist()):
   prob=o['p1'][:,a]*o['p2'][:,b];valid+=float(prob.mean())
   if bool((prob>0).any()):covered.add(j)
  pairs=modes[targets];nll=0.
  for i in range(C):
   nll-=.5*math.log(max(1e-9,float(o['p1'][i,pairs[i,0,0]]*o['p2'][i,pairs[i,0,1]])))
   nll-=.5*math.log(max(1e-9,float(o['p1'][i,pairs[i,1,0]]*o['p2'][i,pairs[i,1,1]])))
  return len(covered)/M,0.,valid,math.log(VOC)*2,nll/C
 return 1/M,0.,1.,0.,10.
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  selection={'selected_K':16,'rule':'predeclared max coverage/quality codebook; K<=8 gate retained; no fresh data used'};(ART/'development_selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps(selection));return
 rows=[]
 for w in FRESH:
  for s in SEEDS:
   modes,targets=data(w,s)
   for method in ['plan_vq','continuous','independent','greedy']:
    ks=KS if method=='plan_vq' else [16]
    for k in ks:
     blob=payload(method,modes,targets,k);o=torch.load(io.BytesIO(blob),weights_only=False);cov,miss,valid,bits,nll=metrics(o,modes,targets);p=PAY/f'{w}_{s}_{method}_K{k}.pt';p.write_bytes(blob)
     rows.append({'world':w,'seed':s,'method':method,'K':k,'joint_mode_coverage':cov,'uncovered_context_fraction':miss,'valid_packet_rate':valid,'bits_proxy':bits,'joint_nll':nll,'payload_bytes':len(blob),'bytes_per_context':len(blob)/C,'hash':hashlib.sha256(blob).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
