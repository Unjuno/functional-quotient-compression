#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';D=4
ROLES=[0.,.2,.4,.6,.8,1.]
HELD=[[0,1],[0,3],[1,0],[1,2],[2,1],[2,3],[3,0],[3,2]]
PATHS=[(0,1),(1,2),(2,3),(3,0)];DEV=[45200,45201];FRESH=[45210,45211,45212];SEEDS=[0,1,2]
def mods(w):
 g=torch.Generator().manual_seed(w+452901);aa=[];bb=[]
 for bank in (aa,bb):
  for _ in range(4):
   q,_=torch.linalg.qr(torch.randn(D,D,generator=g));bank.append(.85*torch.eye(D)+.15*q)
 return torch.stack(aa),torch.stack(bb)
def rot(a):
 c=math.cos(a);s=math.sin(a);r=torch.eye(D);r[0,0]=c;r[0,1]=-s;r[1,0]=s;r[1,1]=c;return r
def getep(w,s,tid,held=True):
 combos=HELD if held else [[p,r] for p in range(4) for r in range(6) if [p,r] not in HELD]
 pid,rid=combos[(tid-(3000 if held else 100))%len(combos)];a=ROLES[rid]
 g=torch.Generator().manual_seed(w*1000033+s*991+tid*73);x=torch.randn(32,D,generator=g);qx=torch.randn(256,D,generator=torch.Generator().manual_seed(w*99173+s*211+tid*31));A,B=mods(w);W=A[PATHS[pid][0]]@rot(a)@B[PATHS[pid][1]]
 return pid,rid,a,x,x@W,qx,qx@W,A,B
def forward(A,B,p,a,x):return x@(A[PATHS[p][0]]@rot(a)@B[PATHS[p][1]])
def score(pred,y):return float(((pred-y).square().mean().sqrt())/(y.square().mean().sqrt()+1e-12))
def eval_one(w,s,tid,defaults):
 p,rid,a,x,y,qx,qy,A,B=getep(w,s,tid)
 out={};t=time.perf_counter();out['pathnet']={'nrmse':score(forward(A,B,p,defaults[p],qx),qy),'angle':defaults[p],'wall':time.perf_counter()-t}
 for name in ['mirror','routing']:
  t=time.perf_counter();losses=[float((forward(A,B,p,z,x)-y).square().mean()) for z in ROLES];idx=min(range(6),key=lambda i:losses[i]);ang=ROLES[idx];out[name]={'nrmse':score(forward(A,B,p,ang,qx),qy),'angle':ang,'role_id':idx,'wall':time.perf_counter()-t}
 t=time.perf_counter();W=torch.linalg.lstsq(x,y).solution;out['flat']={'nrmse':score(qx@W,qy),'W':W,'wall':time.perf_counter()-t};return {'path_id':p,'role_id':rid,'teacher_angle':a,'out':out,'A':A,'B':B}
def serialize(method,items,defaults):
 obj={'format':'ma452-v1','method':method}
 if method in ('pathnet','mirror'):
  obj.update(A=items[0]['A'],B=items[0]['B'],paths=torch.tensor([x['path_id'] for x in items],dtype=torch.uint8))
  if method=='pathnet':obj['default_role_by_path']=torch.tensor(defaults,dtype=torch.float32)
  else:obj['mirror_role_coordinate']=torch.tensor([x['out']['mirror']['angle'] for x in items],dtype=torch.float32)
 elif method=='routing':obj.update(A=items[0]['A'],B=items[0]['B'],paths=torch.tensor([x['path_id'] for x in items],dtype=torch.uint8),role_indices=torch.tensor([x['out']['routing']['role_id'] for x in items],dtype=torch.uint8))
 else:obj['full_task_matrices']=torch.stack([x['out']['flat']['W'] for x in items])
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);args=ap.parse_args();ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if args.phase=='development':
  vals={p:[] for p in range(4)}
  for w in DEV:
   for s in SEEDS:
    for tid in range(100,164):
     p,r,a,x,y,qx,qy,A,B=getep(w,s,tid,held=False)
     for ang in ROLES:vals[p].append((ang,float((forward(A,B,p,ang,qx)-qy).square().mean())))
  defaults=[min(ROLES,key=lambda a:statistics.mean(v for x,v in vals[p] if x==a)) for p in range(4)]
  (ART/'development_selection.json').write_text(json.dumps({'path_default_role':defaults,'rule':'minimum mean dev query MSE per path','fresh_untouched':True},indent=2)+'\n');print(json.dumps({'defaults':defaults}));return
 defaults=json.loads((ART/'development_selection.json').read_text())['path_default_role'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   items=[eval_one(w,s,tid,defaults) for tid in range(3000,3064)]
   for method in ['pathnet','mirror','routing','flat']:
    for N in [1,20,64]:
     batch=items[:N];data=serialize(method,batch,defaults);path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(data)
     for p in range(4):
      group=[x for x in batch if x['path_id']==p]
      if not group:continue
      rows.append({'world':w,'seed':s,'method':method,'path_id':p,'tasks':len(group),'n':N,'heldout_path_role_count':sum(x['role_id']>=0 for x in group),'nrmse_mean':statistics.mean(x['out'][method]['nrmse'] for x in group),'payload_bytes':len(data),'bytes_per_task':len(data)/N,'physical_modules':8 if method in ('pathnet','mirror','routing') else 2*N,'role_search_MAC_per_task':6*(2*D**3+32*D**2) if method in ('mirror','routing') else 0,'query_wall_seconds_mean':statistics.mean(x['out'][method]['wall'] for x in group),'teacher_angles':json.dumps([x['teacher_angle'] for x in group]),'inferred_angles':json.dumps([x['out'][method]['angle'] for x in group]) if method in ('mirror','routing') else '', 'hash':hashlib.sha256(data).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows)}))
if __name__=='__main__':main()
