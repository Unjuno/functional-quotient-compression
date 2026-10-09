import csv,hashlib,io,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[50200,50201];FRESH=[50210,50211,50212];SEEDS=[0,1,2];T=64;D=64;R=4;K=8
g=torch.Generator().manual_seed(502001);TRUE=torch.linalg.qr(torch.randn(D,R,generator=g)).Q;PROT=torch.randn(K,R,generator=g)*.2
def world(w,s):
 g=torch.Generator().manual_seed(w*100003+s*7919+41);labels=torch.randint(0,K,(T,),generator=g);coords=PROT[labels]+torch.randn(T,R,generator=g)*.003;y=coords@TRUE.T;return y,labels

def fit():
 ys=torch.cat([world(w,s)[0] for w in DEV for s in SEEDS]);mean=ys.mean(0);_,_,v=torch.linalg.svd(ys-mean,full_matrices=False);basis=v[:R].T.contiguous();coords=(ys-mean)@basis;g=torch.Generator().manual_seed(502008);cb=coords[torch.randperm(len(coords),generator=g)[:K]].clone()
 for _ in range(50):
  a=torch.cdist(coords,cb).argmin(1);n=cb.clone()
  for j in range(K):
   m=a==j
   if m.any():n[j]=coords[m].mean(0)
  cb=n
 torch.save({'mean':mean,'basis':basis,'codebook':cb},ART/'development.pt')
def payload(m,y,state):
 c=(y-state['mean'])@state['basis'];ids=torch.cdist(c,state['codebook']).argmin(1)
 if m=='dense':o={'method':m,'targets':y}
 elif m=='loreft_fp16':o={'method':m,'mean':state['mean'],'basis':state['basis'],'coords':c.half(),'T':T}
 elif m=='shared_fp16':o={'method':m,'mean':state['mean'],'basis':state['basis'],'coords':c.half(),'T':T}
 else:o={'method':'mirror_vq','mean':state['mean'],'basis':state['basis'],'codebook':state['codebook'],'ids':ids.to(torch.uint8),'T':T}
 b=io.BytesIO();torch.save(o,b);return b.getvalue()
def decode(o):
 if o['method']=='dense':return o['targets']
 if o['method'] in ('loreft_fp16','shared_fp16'):return o['mean']+o['coords'].float()@o['basis'].T
 return o['mean']+o['codebook'][o['ids'].long()]@o['basis'].T
def run(phase):
 ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if phase=='development':fit();print('development basis/codebook fitted');return
 st=torch.load(ART/'development.pt',weights_only=False);rows=[]
 for w in FRESH:
  for s in SEEDS:
   y,labels=world(w,s)
   for m in ['dense','loreft_fp16','shared_fp16','mirror_vq']:
    b=payload(m,y,st);o=torch.load(io.BytesIO(b),weights_only=False);rec=decode(o);err=float((rec-y).norm()/(y.norm()+1e-12));retr=1. if m!='mirror_vq' else float((o['ids'].long()==labels).float().mean());p=PAY/f'{w}_{s}_{m}.pt';p.write_bytes(b);rows.append({'world':w,'seed':s,'method':m,'normalized_rmse':err,'mode_retrieval_accuracy':retr,'payload_bytes':len(b),'bytes_per_task':len(b)/T,'hash':hashlib.sha256(b).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
