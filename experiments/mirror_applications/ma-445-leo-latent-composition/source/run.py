#!/usr/bin/env python3
"""MA-445: compose known latent skill codes on unseen additive pairs."""
import argparse, hashlib, io, json, random, time
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'; PAY=OUT/'payloads'
DEV=[44520,44521]; FRESH=[44530,44531,44532]; SEEDS=[0,1,2]
PAIRS=[(0,1),(0,4),(1,3),(2,5),(3,4),(4,5)]
TRAIN_PAIRS=[(0,2),(0,3),(0,5),(1,2),(1,4),(1,5),(2,3),(2,4),(3,5)]
STEPS=[0,1,3,5]; D=8; K=6; INNER_LR=.05; UPDATES=240; BATCH=8

def fix(s): random.seed(s); torch.manual_seed(s); torch.set_num_threads(1)
def skills(w):
 g=torch.Generator().manual_seed(w+900000); s=torch.zeros(K,D); s[:,:2]=torch.randn(K,2,generator=g)*.45; return s
def task_vec(w,pair):
 s=skills(w); return s[pair[0]]+s[pair[1]]
def samples(w,pair,n,off=0):
 g=torch.Generator().manual_seed(w*100003+pair[0]*991+pair[1]*8191+off+n)
 x=torch.randn(n,D,generator=g); return x,x@task_vec(w,pair)
def adapt(base,code,x,y,steps,lr):
 if code is None:return base
 c=code.detach().clone().requires_grad_(True)
 for _ in range(steps):
  loss=(x@(base+c)-y).square().mean(); grad=torch.autograd.grad(loss,c)[0].detach(); c=c-lr*grad
 return c.detach()
def objective(method,base,decoder,skill_codes,pair):
 if method=='shared':return base
 if method=='taskvec':return base+task_vec_from_delta(skill_codes,pair)
 z=skill_codes[pair[0]]+skill_codes[pair[1]]
 return base+ (z if method=='mirror' else decoder@z)
def task_vec_from_delta(delta,pair): return delta[pair[0]]+delta[pair[1]]
def train(w,seed,method,lr):
 fix(w*71+seed*313+sum(map(ord,method)))
 base=nn.Parameter(torch.zeros(D)); delta=nn.Parameter(torch.randn(K,D)*.02) if method=='taskvec' else None
 codes=nn.Parameter(torch.randn(K,2)*.03) if method in ('mirror','leo') else None
 decoder=nn.Parameter(torch.randn(D,2)*.1) if method=='leo' else torch.eye(D,2)
 ps=[base]+([delta] if delta is not None else [])+([codes] if codes is not None else [])+([decoder] if isinstance(decoder,nn.Parameter) else [])
 opt=torch.optim.Adam(ps,lr=lr); start=time.perf_counter()
 for it in range(UPDATES):
  losses=[]
  for j in range(BATCH):
   pair=TRAIN_PAIRS[(it*BATCH+j)%len(TRAIN_PAIRS)]; x,y=samples(w,pair,16,it)
   if method=='shared': v=base
   elif method=='taskvec': v=base+task_vec_from_delta(delta,pair)
   else:
    z=codes[pair[0]]+codes[pair[1]]; v=base+(torch.nn.functional.pad(z,(0,D-2)) if method=='mirror' else decoder@z)
   pred=x@v
   qx,qy=samples(w,pair,32,it+9101); losses.append((qx@v-qy).square().mean() if method!='shared' else (qx@base-qy).square().mean())
  loss=torch.stack(losses).mean(); opt.zero_grad(); loss.backward(); opt.step()
 trained_delta=delta.detach() if delta is not None else (codes.detach() if codes is not None else torch.zeros(K,D))
 dec=decoder.detach() if isinstance(decoder,nn.Parameter) else decoder.detach()
 return base.detach(),trained_delta,dec,time.perf_counter()-start
def fit_independent(w,pair):
 # Per-combination least squares reference; charged as a complete vector.
 x,y=samples(w,pair,128,414); return torch.linalg.lstsq(x,y).solution
def eval_one(w,seed,pair,method,base,skills_,decoder,steps,path=None):
 sx,sy=samples(w,pair,8,seed+311); qx,qy=samples(w,pair,128,seed+771); start=time.perf_counter()
 if method=='shared': v=base
 elif method=='taskvec': v=base+task_vec_from_delta(skills_,pair)
 else:
   z=skills_[pair[0]]+skills_[pair[1]]; v=base+(torch.nn.functional.pad(z,(0,D-2)) if method=='mirror' else decoder@z)
 if steps:
  # Refinement acts only on a compact 2D coordinate for learned codes; direct delta is frozen.
  if method in ('mirror','leo'):
   z=(skills_[pair[0]]+skills_[pair[1]]).clone(); z=z.detach().requires_grad_(True)
   for _ in range(steps):
    vv=base+(torch.nn.functional.pad(z,(0,D-2)) if method=='mirror' else decoder@z); loss=(sx@vv-sy).square().mean(); g=torch.autograd.grad(loss,z)[0].detach(); z=z-INNER_LR*g
   v=base+(torch.nn.functional.pad(z.detach(),(0,D-2)) if method=='mirror' else decoder@z.detach())
 with torch.no_grad(): err=((qx@v-qy).square().mean().sqrt()/(qy.square().mean().sqrt()+1e-12)).item()
 wall=time.perf_counter()-start
 # Serialize complete per-combination inference state, including skill bank/decoder where used.
 obj={'method':method,'world':w,'seed':seed,'pair':pair,'base':base,'format':'ma445-v1'}
 if method in ('mirror','leo'): obj.update(skill_codes=skills_,decoder=decoder if method=='leo' else torch.empty(0))
 if method=='taskvec':obj['task_deltas']=skills_
 b=io.BytesIO(); torch.save(obj,b); payload=b.getvalue()
 if path: path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload); rel=str(path.relative_to(ROOT.parents[2]))
 else: rel=''
 return err,len(payload),hashlib.sha256(payload).hexdigest(),rel,wall
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);args=ap.parse_args()
 OUT.mkdir(exist_ok=True); PAY.mkdir(exist_ok=True); worlds=DEV if args.phase=='development' else FRESH
 methods=['shared','taskvec','mirror','leo']; lrs=[.01,.03]; rows=[]; selection={}
 for w in worlds:
  for seed in SEEDS:
   for method in methods:
    lrset=lrs if args.phase=='development' else [json.loads((OUT/'development_selection.json').read_text())['lrs'][method]]
    for lr in lrset:
     base,sc,dec,metawall=train(w,seed,method,lr)
     pairs=TRAIN_PAIRS[:3] if args.phase=='development' else PAIRS
     errs=[]
     for pi,pair in enumerate(pairs):
      reps=range(3) if args.phase=='fresh' else range(1)
      for rep in reps:
       for step in STEPS:
        path=PAY/f'{args.phase}_{w}_{seed}_{pair[0]}-{pair[1]}_r{rep}_{method}_{step}.pt' if args.phase=='fresh' else None
        e,b,h,rel,wall=eval_one(w,seed,pair,method,base,sc,dec,step,path); errs.append(e)
        rows.append({'condition':args.phase,'world':w,'seed':seed,'pair':f'{pair[0]}-{pair[1]}','replicate':rep,'method':method,'outer_lr':lr,'steps':step,'nrmse':e,'payload_bytes':b,'hash':h,'path':rel,'adaptation_mac':step*(2 if method in ('mirror','leo') else D)*D*2,'query_wall_seconds':wall,'meta_wall_seconds':metawall})
    if args.phase=='development':
     # Selection recomputed per method across each LR's recorded task errors below.
     pass
 if args.phase=='development':
  for m in methods:
   subset=[r for r in rows if r['method']==m]
   means={lr:sum(r['nrmse'] for r in subset if r['outer_lr']==lr)/max(1,sum(r['outer_lr']==lr for r in subset)) for lr in lrs}
   selection[m]=min(lrs,key=lambda x:means[x])
  (OUT/'development_selection.json').write_text(json.dumps({'lrs':selection,'rule':'lowest mean held-out development NRMSE per method; fresh not accessed','fresh_not_accessed':True},indent=2)+'\n')
 (OUT/f'{args.phase}_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 print(json.dumps({'phase':args.phase,'rows':len(rows),'selection':selection},indent=2))
if __name__=='__main__':main()
