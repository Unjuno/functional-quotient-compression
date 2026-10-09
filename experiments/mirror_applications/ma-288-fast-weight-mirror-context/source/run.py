"""MA-288: context-conditioned associative memory with fast and compressed state."""
import argparse,csv,hashlib,json,math,struct,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];C,M,R,NCTX=8,16,2,16;DEV,FRESH=[28800,28801],[28810,28811,28812];SEEDS=[0,1,2];TRAIN_IDS=list(range(12));HELD_IDS=list(range(12,16));UPDATES=600
torch.set_num_threads(2)
def make(world,seed):
 g=torch.Generator().manual_seed(world*100003+seed*7919+288);keys=F.normalize(torch.randn(NCTX,C,generator=g),dim=1);u=torch.randn(C,R,generator=g);v=torch.randn(R,M,generator=g);rule=u@v;values=keys@rule
 # Contexts are sampled live with noise; complete context identities held out.
 def samples(ids,n,off):
  q=torch.Generator().manual_seed(world*100003+seed*7919+off);idx=torch.tensor(ids)[torch.randint(len(ids),(n,),generator=q)];ctx=keys[idx]+.03*torch.randn(n,C,generator=q);y=ctx@rule;return ctx,idx,y
 train=samples(TRAIN_IDS,64*len(TRAIN_IDS),31);query=samples(HELD_IDS,128*len(HELD_IDS),47)
 return keys,values,rule,train,query
def pack(parts,method):
 flat=torch.cat([p.flatten() for p in parts]).float().numpy().tobytes();meta=json.dumps({'method':method,'shapes':[list(p.shape) for p in parts],'contexts':NCTX},sort_keys=True,separators=(',',':')).encode();return b'MA288\0'+struct.pack('<I',len(meta))+meta+flat
def nrmse(p,y):return float(torch.linalg.vector_norm(p-y)/torch.linalg.vector_norm(y).clamp_min(1e-12))
def run(phase):
 rows=[]
 for world in (DEV if phase=='development' else FRESH):
  for seed in SEEDS:
   keys,values,rule,(ctx,idx,y),(qctx,qidx,qy)=make(world,seed)
   for method in ['no_memory','fast_outer','explicit_updates','static_id','mirror_code','generic_lowrank','generic_mlp']:
    start=time.perf_counter();parts=[];update_macs=0
    if method=='no_memory':pred=torch.zeros_like(qy);parts=[keys]
    elif method=='fast_outer':
     mem=torch.zeros(M,C);mem=mem+torch.einsum('nc,nm->mc',ctx,y)/len(ctx);pred=qctx@mem.T;parts=[mem];update_macs=len(ctx)*C*M
    elif method=='explicit_updates':
     table=torch.zeros(NCTX,M);table.index_add_(0,idx,y);counts=torch.bincount(idx,minlength=NCTX).clamp_min(1);table=table/counts[:,None];pred=table[qidx];parts=[table]
    elif method=='static_id':
     table=nn.Parameter(torch.zeros(NCTX,M));opt=torch.optim.Adam([table],lr=.06)
     for _ in range(UPDATES):opt.zero_grad();loss=F.mse_loss(table[idx],y);loss.backward();opt.step()
     pred=table.detach()[qidx];parts=[table.detach()]
    elif method in ('mirror_code','generic_lowrank'):
     net=nn.Sequential(nn.Linear(C,R,bias=False),nn.Linear(R,M,bias=False));opt=torch.optim.Adam(net.parameters(),lr=.02)
     for _ in range(UPDATES):opt.zero_grad();loss=F.mse_loss(net(ctx),y);loss.backward();opt.step()
     pred=net(qctx).detach();parts=[p.detach() for p in net.parameters()]
    else:
     net=nn.Sequential(nn.Linear(C,32),nn.Tanh(),nn.Linear(32,M));opt=torch.optim.Adam(net.parameters(),lr=.01)
     for _ in range(UPDATES):opt.zero_grad();loss=F.mse_loss(net(ctx),y);loss.backward();opt.step()
     pred=net(qctx).detach();parts=[p.detach() for p in net.parameters()]
    sec=time.perf_counter()-start;err=nrmse(pred,qy);blob=pack(parts,method);path=ROOT/'artifacts'/'payloads'/f'{world}_{seed}_{method}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    rows.append({'phase':phase,'world':world,'seed':seed,'split':'heldout_contexts','method':method,'nrmse':err,'payload_bytes':len(blob),'update_macs':update_macs,'query_macs':len(qctx)*C*M if method=='fast_outer' else 0,'train_seconds':sec,'sha256':hashlib.sha256(blob).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with (ROOT/f'{phase.upper()}_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);run(p.parse_args().phase)
