import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[49800,49801];FRESH=[49820,49821,49822];SEEDS=[0,1,2];E=32;L=8;PS=[0,.02,.05,.1,.2];TRIALS=4000
def gen(seed):
 g=torch.Generator().manual_seed(seed);codes=[];seen=set()
 while len(codes)<E:
  x=torch.randint(0,2,(L,),generator=g,dtype=torch.uint8);key=bytes(x.tolist())
  if key not in seen:seen.add(key);codes.append(x)
 return torch.stack(codes)
def mind(c):
 d=torch.cdist(c.float(),c.float(),p=1);d.fill_diagonal_(999);return int(d.min())
def select():
 opts=[]
 for seed in range(498000,498200):
  c=gen(seed)
  if len({bytes(x.tolist()) for x in c})==E:opts.append((mind(c),seed,c))
 best=max(opts,key=lambda z:(z[0],-z[1]));(ART/'pool_summary.json').write_text(json.dumps({'candidate_count':len(opts),'selected_seed':best[1],'selected_min_distance':best[0]},indent=2)+'\n');return best[1]
def nearest(c,obs):return (obs[:,None,:]!=c[None,:,:]).sum(2).argmin(1)
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  seed=select();(ART/'selection.json').write_text(json.dumps({'seed':seed},indent=2)+'\n');print(seed);return
 seed=json.loads((ART/'selection.json').read_text())['seed'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   for method,c in [('random',gen(499001)),('distance_optimized',gen(seed)),('binary5',torch.tensor([[(i>>j)&1 for j in range(4,-1,-1)]+[0,0,0] for i in range(E)],dtype=torch.uint8))]:
    for p in PS:
     g=torch.Generator().manual_seed(w*100003+s*7919+int(p*1000)+c.shape[1]);labels=torch.arange(E).repeat((TRIALS+E-1)//E)[:TRIALS];obs=c[labels]^(torch.rand(c[labels].shape,generator=g)<p).to(torch.uint8);dec=nearest(c,obs);acc=float((dec==labels).float().mean());b=io.BytesIO();torch.save({'codes':c,'expert_map':torch.arange(E,dtype=torch.uint8),'bits':c.shape[1]},b);blob=b.getvalue();Lx=c.shape[1];path=PAY/f'{w}_{s}_{method}_L{Lx}_p{p}.pt';path.write_bytes(blob);rows.append({'world':w,'seed':s,'method':method,'length_bits':Lx,'flip_probability':p,'accuracy':acc,'misroute_rate':1-acc,'min_distance':mind(c),'payload_bytes':len(blob),'decode_ops':E*Lx*TRIALS,'hash':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
