import argparse,csv,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,BlockViews,ROLES,D,BLOCK,NBLK,block_matrix,shift_codes,compute_proxy

ROOT=Path(__file__).resolve().parents[1];UPDATES=1200;BATCH=64
FIELDS=['split','seed','condition','method','learning_rate','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','activation_mse','r2']

def world(seed,condition):
 g=torch.Generator().manual_seed(seed);address_seed=seed+900
 if condition=='aligned_block_shift':
  kernels=torch.randn(NBLK,NBLK,BLOCK,generator=g)*.35
  shifts=shift_codes(address_seed)
  weights=torch.stack([block_matrix(kernels,shifts[r]) for r in range(ROLES)])
  bias=torch.randn(D,generator=g)*.1;biases=bias.expand(ROLES,-1).clone()
 else:
  weights=torch.randn(ROLES,D,D,generator=g)*.35
  biases=torch.randn(ROLES,D,generator=g)*.1
 return {'weights':weights,'biases':biases,'address_seed':address_seed,'kernels':kernels if condition=='aligned_block_shift' else None,'shifts':shifts if condition=='aligned_block_shift' else None}

def targets(x,role,w):return torch.einsum('bi,bij->bj',x,w['weights'][role])+w['biases'][role]

def evaluate(m,w,seed):
 g=torch.Generator().manual_seed(seed);x=torch.randn(4096,D,generator=g);r=torch.randint(ROLES,(4096,),generator=g);y=targets(x,r,w);m.eval()
 with torch.no_grad():
  pred=m(x,r);mse=float((pred-y).square().mean());r2=float(1-(pred-y).square().sum()/(y-y.mean()).square().sum())
 return mse,r2,x,r

def fit(method,w,init_seed,data_seed,lr):
 torch.manual_seed(init_seed);m=BlockViews(method,w['address_seed']);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(data_seed);m.train();t=time.perf_counter()
 for _ in range(UPDATES):
  x=torch.randn(BATCH,D,generator=g);r=torch.randint(ROLES,(BATCH,),generator=g);loss=F.mse_loss(m(x,r),targets(x,r,w));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 return m,time.perf_counter()-t

def throughput(m,x,r):
 x=x[:64];r=r[:64];m.eval()
 with torch.no_grad():
  for _ in range(5):m(x,r)
  t=time.perf_counter()
  for _ in range(30):m(x,r)
 return 64*30/(time.perf_counter()-t)

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','fresh'],required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);rows=[]
 for seed in a.seeds:
  for ci,condition in enumerate(('aligned_block_shift','independent_roles')):
   w=world(seed,condition);data_seed=seed+10000+ci*5000
   for mi,method in enumerate(METHODS):
    m,wall=fit(method,w,seed+mi*73+ci*10000,data_seed,a.lr);mse,r2,x,r=evaluate(m,w,seed+91+ci*3000)
    rows.append({'split':a.split,'seed':seed,'condition':condition,'method':method,'learning_rate':f'{a.lr:.3f}','serialized_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':f'{wall:.6f}','inference_examples_per_s':f'{throughput(m,x,r):.3f}','activation_mse':f'{mse:.10g}','r2':f'{r2:.8g}'})
    print(a.split,seed,condition,method,'MSE',mse,'bytes',rows[-1]['serialized_bytes'],flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');wr.writeheader();wr.writerows(rows)
 print(json.dumps({'split':a.split,'seeds':a.seeds,'learning_rate':a.lr,'rows':len(rows)}))
if __name__=='__main__':main()
