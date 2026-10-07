#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,FactorizedExpertDepth,compute_proxy,rotate
ROOT=Path(__file__).resolve().parents[1];E=4;L=4;D=16;O=12;UPDATES=1200;FIELDS=['condition','world_or_seed','mode','method','serialized_model_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
def make_world(seed,mode):
 g=torch.Generator().manual_seed(seed);w=torch.randn(D,O,generator=g)*.6;b=torch.randn(O,generator=g)*.1;ae=torch.rand(E,4,generator=g)*1.2-.6;al=torch.rand(L,4,generator=g)*1.2-.6
 if mode=='independent_pairs':wi=torch.randn(E,L,D,O,generator=g)*.6;bi=torch.randn(E,L,O,generator=g)*.1
 else:wi=bi=None
 return {'w':w,'b':b,'ae':ae,'al':al,'wi':wi,'bi':bi}
def target(x,e,l,w,mode):
 if mode=='independent_pairs':return torch.einsum('bd,bdo->bo',x,w['wi'][e,l])+w['bi'][e,l]
 h=x.clone();h[:,:8]=rotate(x[:,:8],w['ae'][e]);h[:,8:]=rotate(x[:,8:],w['al'][l]);return h@w['w']+w['b']
def fit(method,w,mode,seed,data_seed,lr):
 torch.manual_seed(seed);m=FactorizedExpertDepth(method,data_seed,E,L,D,O);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed+31);t0=time.perf_counter();m.train()
 for _ in range(UPDATES):
  x=torch.randn(64,D,generator=g);e=torch.randint(E,(64,),generator=g);l=torch.randint(L,(64,),generator=g);y=target(x,e,l,w,mode);pred=m(x,e,l);loss=F.mse_loss(pred,y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t0
def evaluate(m,w,mode,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(4096,D,generator=g);e=torch.randint(E,(4096,),generator=g);l=torch.randint(L,(4096,),generator=g);y=target(x,e,l,w,mode);m.eval()
 with torch.no_grad():pred=m(x,e,l);mse=float((pred-y).square().mean());r2=float(1-(pred-y).square().sum()/(y-y.mean()).square().sum())
 return mse,r2,x,e,l
def throughput(m,x,e,l):
 x=x[:64];e=e[:64];l=l[:64];m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,e,l)
  t=time.perf_counter()
  for _ in range(30):m(x,e,l)
 return 64*30/(time.perf_counter()-t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1);modes=['factorized_axes','independent_pairs']
 if a.phase=='dev':worlds=[(25100,251000)];lrs=[.003,.01]
 else:
  if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen lr')
  worlds=[(25101,251001),(25102,251002),(25103,251003)];lrs=[a.lr]
 rows=[];scores={lr:[] for lr in lrs}
 for wid,init in worlds:
  for mode in modes:
   w=make_world(wid,mode);ds=init+(0 if mode==modes[0] else 10000)
   for lr in lrs:
    for i,method in enumerate(METHODS):
     m,elapsed=fit(method,w,mode,init+i*197+int(lr*10000),ds,lr);mse,r2,x,e,l=evaluate(m,w,mode,init+91)
     if a.phase=='dev':scores[lr].append(mse)
     row={'condition':a.phase,'world_or_seed':wid,'mode':mode,'method':method,'serialized_model_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*64,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*64),'wall_time_s':round(elapsed,6),'inference_examples_per_s':round(throughput(m,x,e,l),3),'primary_metric':'heldout_MSE','primary_value':f'{mse:.10g}','secondary_metric':'R2','secondary_value':f'{r2:.8g}','status_note':f'lr={lr}; matched minibatches; no Cartesian combination capacity claim'};rows.append(row);print(a.phase,wid,mode,method,lr,'MSE',mse,'R2',r2,'bytes',row['serialized_model_bytes'],flush=True)
 if a.phase=='dev':
  best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-251','selected_common_lr':best,'mean_mse_by_lr':{str(k):sum(v)/len(v) for k,v in scores.items()},'fresh_worlds':[25101,25102,25103],'updates':UPDATES},indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
 print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':len(rows)}))
if __name__=='__main__':main()
