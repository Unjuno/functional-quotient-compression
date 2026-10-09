#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';D=4
ANGLES=[0.,.2,.4,.6,.8,1.];PATHS=[(0,1),(1,2),(2,3),(3,0)]
DEV=[45100,45101];FRESH=[45110,45111,45112];SEEDS=[0,1,2]
def modules(w):
 g=torch.Generator().manual_seed(w+451901);aa=[];bb=[]
 for _ in range(4):
  q,_=torch.linalg.qr(torch.randn(D,D,generator=g));aa.append(.85*torch.eye(D)+.15*q)
 for _ in range(4):
  q,_=torch.linalg.qr(torch.randn(D,D,generator=g));bb.append(.85*torch.eye(D)+.15*q)
 return torch.stack(aa),torch.stack(bb)
def rot(a):
 c=math.cos(a);s=math.sin(a);r=torch.eye(D);r[0,0]=c;r[0,1]=-s;r[1,0]=s;r[1,1]=c;return r
def episode(w,seed,tid):
 path_id=(tid-100 if tid<2000 else tid-2000)%4;path=PATHS[path_id]
 g=torch.Generator().manual_seed(w*1000003+seed*997+tid*71);angle=ANGLES[int(torch.randint(len(ANGLES),(1,),generator=g))]
 x=torch.randn(32,D,generator=g);qx=torch.randn(256,D,generator=torch.Generator().manual_seed(w*99991+seed*193+tid*29))
 A,B=modules(w);W=A[path[0]]@rot(angle)@B[path[1]];return path_id,path,angle,x,x@W,qx,qx@W,A,B
def predict(A,B,path,angle,x):return x@(A[path[0]]@rot(angle)@B[path[1]])
def one(w,seed,tid,defaults):
 path_id,path,angle,x,y,qx,qy,A,B=episode(w,seed,tid);out={};
 # PathNet: selected physical modules and a path-level role fixed on development.
 start=time.perf_counter();pp=predict(A,B,path,defaults[path_id],qx);out['pathnet']={'nrmse':float(((pp-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)),'wall':time.perf_counter()-start,'angle':defaults[path_id]}
 # Mirror: same path/module bank, infer one role coordinate from support using frozen grid.
 start=time.perf_counter();candidates=[]
 for a in ANGLES:candidates.append(float((predict(A,B,path,a,x)-y).square().mean()))
 ma=ANGLES[min(range(len(ANGLES)),key=lambda i:candidates[i])];mp=predict(A,B,path,ma,qx);out['mirror']={'nrmse':float(((mp-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)),'wall':time.perf_counter()-start,'angle':ma}
 # No path selection: one fixed physical route and its development default.
 start=time.perf_counter();sp=predict(A,B,PATHS[0],defaults[0],qx);out['shared']={'nrmse':float(((sp-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)),'wall':time.perf_counter()-start,'angle':defaults[0]}
 # Independent per-task full matrix from support least squares.
 start=time.perf_counter();W=torch.linalg.lstsq(x,y).solution;ip=qx@W;out['independent']={'nrmse':float(((ip-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-12)),'wall':time.perf_counter()-start,'angle':None}
 return {'path_id':path_id,'path':path,'teacher_angle':angle,'outputs':out,'A':A,'B':B}
def serial(method,items,defaults):
 A=items[0]['A'];B=items[0]['B'];obj={'format':'ma451-v1','method':method}
 if method in ('pathnet','mirror'):
  obj.update(A_modules=A,B_modules=B,paths=torch.tensor([x['path'] for x in items]))
  if method=='pathnet':obj['path_default_angles']=torch.tensor(defaults)
  else:obj['mirror_angles']=torch.tensor([x['outputs']['mirror']['angle'] for x in items])
 elif method=='shared':obj.update(A_module=A[PATHS[0][0]],B_module=B[PATHS[0][1]],angle=float(defaults[0]))
 else:obj['full_weights']=torch.empty(0)
 if method=='independent':
  # Store the fitted per-task matrix so inference is fully reconstructible.
  weights=[]
  for x in items:
   _,_,_,sx,sy,_,_,_,_=episode(x['world'],x['seed'],x['tid']);weights.append(torch.linalg.lstsq(sx,sy).solution)
  obj['full_weights']=torch.stack(weights)
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);args=ap.parse_args();ART.mkdir(exist_ok=True);PAY.mkdir(exist_ok=True)
 if args.phase=='development':
  losses={p:[] for p in range(4)}
  for w in DEV:
   for s in SEEDS:
    for tid in range(100,164):
     pid,path,a,x,y,qx,qy,A,B=episode(w,s,tid)
     for cand in ANGLES:losses[pid].append((cand,float((predict(A,B,path,cand,qx)-qy).square().mean())))
  defaults=[]
  for pid in range(4):defaults.append(min(ANGLES,key=lambda c:sum(v for a,v in losses[pid] if a==c)/max(1,sum(a==c for a,v in losses[pid]))))
  (ART/'development_selection.json').write_text(json.dumps({'default_angle_by_path':defaults,'rule':'minimum mean dev query MSE per path','fresh_not_accessed':True},indent=2)+'\n');print(json.dumps({'defaults':defaults}));return
 defaults=json.loads((ART/'development_selection.json').read_text())['default_angle_by_path'];rows=[]
 for w in FRESH:
  for s in SEEDS:
   items=[]
   for tid in range(2000,2064):
    q=one(w,s,tid,defaults);q.update(world=w,seed=s,tid=tid);items.append(q)
   for method in ['pathnet','mirror','shared','independent']:
    for N in [1,20,64]:
     sample=items[:N];data=serial(method,sample,defaults);p=PAY/f'{w}_{s}_{method}_N{N}.pt';p.write_bytes(data)
     for pid in range(4):
      g=[x for x in sample if x['path_id']==pid]
      if not g:continue
      vals=[x['outputs'][method]['nrmse'] for x in g]
      rows.append({'world':w,'seed':s,'method':method,'path_id':pid,'tasks':len(g),'n':N,'nrmse_mean':statistics.mean(vals),'payload_bytes':len(data),'bytes_per_task':len(data)/N,'physical_modules':8 if method in ('pathnet','mirror') else 2 if method=='shared' else N*2,'transform_MAC_per_task':4*D*D if method=='mirror' else 0,'query_wall_seconds_mean':statistics.mean(x['outputs'][method]['wall'] for x in g),'teacher_angles':json.dumps([x['teacher_angle'] for x in g]),'inferred_angles':json.dumps([x['outputs']['mirror']['angle'] for x in g]) if method=='mirror' else '', 'hash':hashlib.sha256(data).hexdigest(),'path':str(p.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows)}))
if __name__=='__main__':main()
