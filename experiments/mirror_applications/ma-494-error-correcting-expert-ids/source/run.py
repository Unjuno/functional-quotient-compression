import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[49400,49401];FRESH=[49420,49421,49422];SEEDS=[0,1,2];E=32;LENS=[5,7,9,11,15];PS=[0,.02,.05,.1,.2];TRIALS=4000
def make_codes(L,seed):
 g=torch.Generator().manual_seed(seed);codes=[]
 while len(codes)<E:
  c=torch.randint(0,2,(L,),generator=g,dtype=torch.uint8)
  if not codes or min(int((c!=x).sum()) for x in codes)>=max(1,L//4):codes.append(c)
 return torch.stack(codes)
def select():
 scores=[]
 for seed in [494001,494002]:
  vals=[]
  for L in [7,9,11]:vals.append(int(torch.cdist(make_codes(L,seed).float(),make_codes(L,seed).float(),p=1).fill_diagonal_(999).min()))
  scores.append((min(vals),seed))
 return max(scores)[1]
def nearest(codes,obs):return (obs[:,None,:]!=codes[None,:,:]).sum(2).argmin(1)
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':
  seed=select();(ART/'selection.json').write_text(json.dumps({'seed':seed,'minimum_distance_by_length':{str(L):int(torch.cdist(make_codes(L,seed).float(),make_codes(L,seed).float(),p=1).fill_diagonal_(999).min()) for L in [7,9,11]}},indent=2)+'\n');print(seed);return
 seed=json.loads((ART/'selection.json').read_text())['seed'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   for L in LENS:
    base_codes=torch.tensor([[(i>>j)&1 for j in range(4,-1,-1)] for i in range(E)],dtype=torch.uint8)
    if L==5:codes=base_codes
    elif L==15:codes=base_codes.repeat_interleave(3,dim=1)
    else:codes=make_codes(L,seed)
    g=torch.Generator().manual_seed(w*100003+s*7919+L)
    for p in PS:
     labels=torch.arange(E).repeat((TRIALS+E-1)//E)[:TRIALS];clean=codes[labels];noise=(torch.rand(clean.shape,generator=g)<p).to(torch.uint8);obs=clean^noise
     dec=nearest(codes,obs);acc=float((dec==labels).float().mean());blob=io.BytesIO();torch.save({'codes':codes,'expert_map':torch.arange(E,dtype=torch.uint8),'L':L},blob);data=blob.getvalue();name='binary' if L==5 else 'repeat3' if L==15 else 'ecoc';path=PAY/f'{w}_{s}_{name}_L{L}_p{p}.pt';path.write_bytes(data)
     rows.append({'world':w,'seed':s,'length_bits':L,'method':name,'flip_probability':p,'accuracy':acc,'misroute_rate':1-acc,'payload_bytes':len(data),'bits_per_address':L,'min_distance':int(torch.cdist(codes.float(),codes.float(),p=1).fill_diagonal_(999).min()),'decode_hamming_ops':E*L*TRIALS,'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
