#!/usr/bin/env python3
"""MA-371 synthetic MatFormer nested-width / unseen-mix mechanism screen."""
import argparse,csv,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
WIDTHS=(2,4,8); D=8; CLASSES=4; NTRAIN=NTEST=1024; UPDATES=1000; LR=.035
TRAIN_CONFIGS=((2,2),(4,4),(8,8)); MIXES=((2,8),(4,2),(8,4))

def payload(arrays):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
  for n in sorted(arrays):
   q=io.BytesIO();np.save(q,arrays[n],allow_pickle=False);zi=zipfile.ZipInfo(n+'.npy',(1980,1,1,0,0,0));z.writestr(zi,q.getvalue())
 return b.getvalue()

def softmax(a):
 a=a-a.max(axis=1,keepdims=True);e=np.exp(a);return e/e.sum(axis=1,keepdims=True)
def metrics(logits,y):
 p=softmax(logits);return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),float((p.argmax(1)==y).mean())
def runworld(seed):
 rng=np.random.default_rng(seed);x=rng.normal(size=(NTRAIN+NTEST,D)).astype(np.float64); teacher=rng.normal(size=(D,CLASSES)); y=(x@teacher).argmax(axis=1).astype(np.int64); xtr,xte=x[:NTRAIN],x[NTRAIN:];ytr,yte=y[:NTRAIN],y[NTRAIN:]
 # Shared layer weights, nested hidden widths. Both layers have width 8 physical storage.
 W1=[rng.normal(0,.18,(D,8)),rng.normal(0,.18,(8,8))];W2=[rng.normal(0,.18,(8,CLASSES)) for _ in range(2)]
 # Learnable random linear teacher; optimize shared nested FFN and width/layer gains.
 torch.manual_seed(seed); device='cpu'; xt=torch.tensor(xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long)
 tW0=torch.nn.Parameter(torch.randn(D,8)*.25);tW1=torch.nn.Parameter(torch.randn(8,8)*.25);tW2=torch.nn.Parameter(torch.randn(8,CLASSES)*.25);tg=torch.nn.Parameter(torch.ones(3,2,8))
 opt=torch.optim.SGD([tW0,tW1,tW2,tg],lr=LR)
 start=time.perf_counter()
 for _ in range(UPDATES):
  opt.zero_grad(); loss=0
  for gi,w in enumerate(WIDTHS):
   h=torch.relu(xt@tW0[:,:w])*tg[gi,0,:w]
   h=torch.relu(h@tW1[:w,:w])*tg[gi,1,:w]
   loss=loss+torch.nn.functional.cross_entropy(h@tW2[:w,:],yt)
  loss.backward();opt.step()
 wall=time.perf_counter()-start
 W1=[tW0.detach().numpy(),tW1.detach().numpy()];W2=[rng.normal(0,.1,(D,CLASSES)),tW2.detach().numpy()];g=tg.detach().numpy()
 rows=[];arr={"W1_0":W1[0].astype(np.float32),"W2_0":W2[0].astype(np.float32),"W1_1":W1[1].astype(np.float32),"W2_1":W2[1].astype(np.float32)}
 # Physical native shared prefix model bytes.
 shared_b=len(payload(arr))
 # Assemble actual coefficient payload and independent function bank.
 coeff_arr={**arr,"gains":g.astype(np.float32)}; direct_b=len(payload(coeff_arr)); mirror_b=direct_b
 # Independent FFN bank is charged per width per layer.
 ind={}
 for gi,w in enumerate(WIDTHS):
  for li in range(2):
   ind[f"W1_{gi}_{li}"]=(W1[li][:,:w] if li==0 else (W1[li][:WIDTHS[gi],:w]*g[gi,li,:w])).astype(np.float32)
   ind[f"W2_{gi}_{li}"]=(W2[1][:w,:]*g[gi,1,:w,None]).astype(np.float32) if li==1 else W2[li][:w,:].astype(np.float32)
 independent_b=len(payload(ind))
 for cfg in TRAIN_CONFIGS+MIXES:
  gi=WIDTHS.index(cfg[0]) if cfg[0]==cfg[1] else -1
  for method in ("shared_prefix","direct_coeff","mirror_gate","independent"):
   h=xte
   for li,w in enumerate(cfg):
    if method in ('direct_coeff','mirror_gate') and gi>=0: gain=g[gi,li,:w]
    elif method=='independent' and gi>=0: gain=np.ones(w)
    elif method in ('direct_coeff','mirror_gate') and gi<0:
     # Unseen mixed granularity uses factorized layer x width interpolation code.
     wi=WIDTHS.index(w);gain=g[wi,li,:w]
    elif method=='independent' and gi<0:
     wi=WIDTHS.index(w);gain=np.ones(w)
    else: gain=np.ones(w)
    h=np.tanh(h@(W1[li][:,:w] if li==0 else W1[li][:cfg[0],:w]))*gain
   logits=h@W2[1][:cfg[1],:]
   nll,acc=metrics(logits,yte)
   b={'shared_prefix':shared_b,'direct_coeff':direct_b,'mirror_gate':mirror_b,'independent':independent_b}[method]
   macs=NTEST*sum(D*w+2*w*CLASSES for w in cfg)
   rows.append({'world':seed,'method':method,'configuration':f'{cfg[0]}x{cfg[1]}','serialized_bytes':b,'train_examples':NTRAIN,'optimizer_updates':UPDATES if method in ('direct_coeff','mirror_gate') else 0,'mac_proxy':macs,'wall_time_s':f'{wall:.6f}','nll':f'{nll:.8g}','accuracy':f'{acc:.8g}','status_note':'development; heldout mixes not optimized explicitly'})
 return rows,{'shared':shared_b,'direct':direct_b,'mirror':mirror_b,'independent':independent_b}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'artifacts'/'dev_results.csv'));a=ap.parse_args();rows=[];sizes=[]
 for seed in (37101,37102):
  r,s=runworld(seed);rows+=r;sizes.append(s)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'sizes':sizes},indent=2))
if __name__=='__main__':main()
