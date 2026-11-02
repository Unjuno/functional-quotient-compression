import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[50100,50101];FRESH=[50120,50121,50122];SEEDS=[0,1,2];T=64;D=64;R=4;KS=[8,16,32,64]
g=torch.Generator().manual_seed(501001);TRUE=torch.linalg.qr(torch.randn(D,R,generator=g)).Q
def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);c=torch.randn(T,R,generator=g);y=c@TRUE.T+torch.randn(T,D,generator=g)*.015;return y

def fit_basis():
 ys=torch.cat([world(w,s) for w in DEV for s in SEEDS]);mean=ys.mean(0);_,_,v=torch.linalg.svd(ys-mean,full_matrices=False);basis=v[:R].T.contiguous();coords=(ys-mean)@basis
 books={}
 for k in KS:
  g=torch.Generator().manual_seed(501000+k);cb=coords[torch.randperm(len(coords),generator=g)[:k]].clone()
  for _ in range(40):
   a=torch.cdist(coords,cb).argmin(1);n=cb.clone()
   for j in range(k):
    m=a==j
    if m.any():n[j]=coords[m].mean(0)
   cb=n
  books[str(k)]=cb
 torch.save({'mean':mean,'basis':basis,'books':books},ART/'development.pt')

def pack(method,y,state,k=0):
 mean=state['mean'];basis=state['basis'];coords=(y-mean)@basis;rec=mean+coords@basis.T
 if method=='dense':o={'method':method,'targets':y.clone(),'T':T}
 elif method=='loreft_fp16':o={'method':method,'mean':mean,'basis':basis,'coords':coords.half(),'T':T}
 elif method=='shared_only':o={'method':method,'mean':mean,'basis':basis,'T':T}
 else:
  cb=state['books'][str(k)];ids=torch.cdist(coords,cb).argmin(1);o={'method':'mirror_vq','mean':mean,'basis':basis,'codebook':cb,'ids':ids.to(torch.uint8),'K':k,'T':T}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()
def decode(o):
 if o['method']=='dense':return o['targets']
 y=o['mean'][None,:].expand(o['T'],-1).clone()
 if o['method']=='loreft_fp16':return y+o['coords'].float()@o['basis'].T
 if o['method']=='shared_only':return y
 return y+o['codebook'][o['ids'].long()]@o['basis'].T
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':fit_basis();print('basis and codebooks fitted on development');return
 state=torch.load(ART/'development.pt',weights_only=False);rows=[]
 for w in FRESH:
  for s in SEEDS:
   y=world(w,s)
   for m in ['dense','loreft_fp16','shared_only','mirror_vq']:
    for k in (KS if m=='mirror_vq' else [0]):
     blob=pack(m,y,state,k);o=torch.load(io.BytesIO(blob),weights_only=False);rec=decode(o);err=float((rec-y).norm()/(y.norm()+1e-12));p=PAY/f'{w}_{s}_{m}_K{k}.pt';p.write_bytes(blob);rows.append({'world':w,'seed':s,'method':m,'K':k,'normalized_rmse':err,'payload_bytes':len(blob),'bytes_per_task':len(blob)/T,'application_MAC_per_token':D*R if m in ['loreft_fp16','mirror_vq'] else 0,'hash':hashlib.sha256(blob).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
