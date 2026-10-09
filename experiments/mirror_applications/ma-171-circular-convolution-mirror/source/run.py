from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS, ExpertViews, address_matrices, compute_proxy

ROOT=Path(__file__).resolve().parents[1]
ROLES,D,O,UPDATES,BATCH=4,16,12,1200,64
FIELDS=['split','seed','condition','method','learning_rate','serialized_bytes','train_examples','optimizer_updates','active_compute_proxy','wall_time_s','inference_examples_per_s','activation_mse','r2']

def world(seed, condition):
    g=torch.Generator().manual_seed(seed)
    w=torch.randn(D,O,generator=g)*.6; b=torch.randn(O,generator=g)*.1
    address_seed=seed+900
    if condition=='aligned_hrr':
        mats=address_matrices(address_seed,'hrr',ROLES,D)
        wi=torch.stack([m@w for m in mats]);bi=b.expand(ROLES,-1).clone()
    else:
        wi=torch.randn(ROLES,D,O,generator=g)*.6
        bi=torch.randn(ROLES,O,generator=g)*.1
        mats=None
    return {'w':w,'b':b,'wi':wi,'bi':bi,'address_seed':address_seed,'teacher_mats':mats}

def targets(x, role, w, condition):
    if condition=='aligned_hrr':
        a=w['teacher_mats'][role]
        h=torch.bmm(x[:,None,:],a).squeeze(1)
        return h@w['w']+w['b']
    return torch.einsum('bd,bdo->bo',x,w['wi'][role])+w['bi'][role]

def evaluate(m,w,condition,seed):
    g=torch.Generator().manual_seed(seed);x=torch.randn(4096,D,generator=g);r=torch.randint(ROLES,(4096,),generator=g)
    y=targets(x,r,w,condition);m.eval()
    with torch.no_grad():
        pred=m(x,r);mse=float((pred-y).square().mean());r2=float(1-(pred-y).square().sum()/(y-y.mean()).square().sum())
    return mse,r2,x,r

def fit(method,w,condition,init_seed,data_seed,lr):
    torch.manual_seed(init_seed)
    m=ExpertViews(method,w['address_seed'],ROLES,D,O)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    g=torch.Generator().manual_seed(data_seed);m.train();start=time.perf_counter()
    for _ in range(UPDATES):
        x=torch.randn(BATCH,D,generator=g);r=torch.randint(ROLES,(BATCH,),generator=g)
        y=targets(x,r,w,condition);loss=F.mse_loss(m(x,r),y)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return m,time.perf_counter()-start

def throughput(m,x,r):
    x=x[:64];r=r[:64];m.eval()
    with torch.no_grad():
        for _ in range(5):m(x,r)
        t=time.perf_counter()
        for _ in range(30):m(x,r)
    return 64*30/(time.perf_counter()-t)

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','fresh'],required=True);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);rows=[]
    for seed in a.seeds:
        for ci,condition in enumerate(('aligned_hrr','independent_roles')):
            w=world(seed,condition);data_seed=seed+10000+ci*5000
            for mi,method in enumerate(METHODS):
                m,wall=fit(method,w,condition,seed+mi*73+ci*10000,data_seed,a.lr)
                mse,r2,x,r=evaluate(m,w,condition,seed+91+ci*3000)
                rows.append({'split':a.split,'seed':seed,'condition':condition,'method':method,'learning_rate':f'{a.lr:.3f}','serialized_bytes':m.serialized_payload_bytes(),'train_examples':UPDATES*BATCH,'optimizer_updates':UPDATES,'active_compute_proxy':compute_proxy(method,UPDATES*BATCH),'wall_time_s':f'{wall:.6f}','inference_examples_per_s':f'{throughput(m,x,r):.3f}','activation_mse':f'{mse:.10g}','r2':f'{r2:.8g}'})
                print(a.split,seed,condition,method,'mse',mse,'bytes',rows[-1]['serialized_bytes'],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writeheader()
    with a.out.open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
    print(json.dumps({'split':a.split,'seeds':a.seeds,'lr':a.lr,'rows':len(rows)}))

if __name__=='__main__':main()
