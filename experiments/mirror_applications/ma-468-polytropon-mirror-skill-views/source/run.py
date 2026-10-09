#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[46800,46801];FRESH=[46820,46821,46822];SEEDS=[0,1,2];PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];ANGLES=torch.linspace(-math.pi,math.pi,361)
def rot(a):
 c,s=torch.cos(a),torch.sin(a);return torch.stack([c,-s,s,c]).reshape(2,2)
def make(w,s):
 g=torch.Generator().manual_seed(w*1000003+s*997+53);b1=torch.randn(2,generator=g)*.4+torch.tensor([.8,.2]);b2=torch.randn(2,generator=g)*.4+torch.tensor([-.2,.8]);theta=torch.tensor(1.05);delta=torch.tensor([.55,-.35]);V=torch.stack([b1,rot(theta)@b1,b2,b2+delta]);xs=[];ys=[];qs=[];qys=[]
 for k in range(4):
  x=torch.randn(64,2,generator=torch.Generator().manual_seed(w*99131+s*71+k*23));q=torch.randn(256,2,generator=torch.Generator().manual_seed(w*1217+s*79+k*31));xs.append(x);ys.append(x@V[k]);qs.append(q);qys.append(q@V[k])
 return V,xs,ys,qs,qys
def fitvec(x,y):return torch.linalg.solve(x.T@x+torch.eye(2)*1e-6,x.T@y)
def fit(w,s):
 V,xs,ys,qs,qys=make(w,s);begin=time.perf_counter();base=torch.stack([fitvec(xs[k],ys[k]) for k in range(4)]); fitwall=time.perf_counter()-begin
 # Base modules calibrated from canonical skills 0 and 2 only.
 b1=base[0];b2=base[2];start=time.perf_counter();
 def bestangle(target,base):
  losses=[float(((xs[target]@(rot(a)@base)-ys[target]).square().mean())) for a in ANGLES];i=min(range(len(losses)),key=losses.__getitem__);return ANGLES[i],losses[i]
 a1,_=bestangle(1,b1);a3,_=bestangle(3,b2);viewwall=time.perf_counter()-start
 # A private residual is obtained from support and charged separately.
 residual=base[3]-b2
 modules={'no_view':base[[0,2]],'mirror':torch.stack([b1,b2]),'mirror_private':torch.stack([b1,b2]),'full':base}
 angles=torch.tensor([0.,float(a1),0.,float(a3)])
 pred={'no_view':torch.stack([b1,b1,b2,b2]),'mirror':torch.stack([b1,rot(a1)@b1,b2,rot(a3)@b2]),'mirror_private':torch.stack([b1,rot(a1)@b1,b2,b2+residual]),'full':base}
 task_q=[]
 for i,(a,b) in enumerate(PAIRS):
  q=torch.randn(256,2,generator=torch.Generator().manual_seed(w*99131+s*17+i*19));target=q@(V[a]+V[b]);task_q.append((q,target))
 return V,modules,angles,residual,pred,task_q,fitwall,viewwall

def nrmse(p,y):return float((p-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-9))
def score(pred,task_q,which):
 errs=[];wall=[]
 for j in which:
  a,b=PAIRS[j];q,y=task_q[j];t=time.perf_counter();o=q@(pred[a]+pred[b]);wall.append(time.perf_counter()-t);errs.append(nrmse(o,y))
 return statistics.mean(errs),statistics.mean(wall)
def package(method,mods,angles,resid,N):
 mapping=[0,1,2,3] if method=='full' else [0,0,1,1]
 obj={'format':'ma468-v1','method':method,'N':N,'allocations':torch.tensor(PAIRS[:N],dtype=torch.uint8),'skill_to_module':torch.tensor(mapping,dtype=torch.uint8)}
 if method=='no_view':obj['modules']=mods[method]
 elif method=='mirror':obj.update({'modules':mods[method],'view_angles':angles})
 elif method=='mirror_private':obj.update({'modules':mods[method],'view_angles':angles,'private_residual_skill3':resid})
 else:obj['modules']=mods[method]
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def run(phase):
 worlds=DEV if phase=='development' else FRESH;rows=[]
 for w in worlds:
  for s in range(3):
   V,mods,angles,resid,pred,tq,fitwall,viewwall=fit(w,s)
   if phase=='development':continue
   for m in ['no_view','mirror','mirror_private','full']:
    for N in [1,3,6]:
     b=package(m,mods,angles,resid,N);path=PAY/f'{w}_{s}_{m}_N{N}.pt';path.write_bytes(b)
     err,qwall=score(pred[m],tq,[4,5] if N==6 else list(range(min(N,6))))
     rows.append({'world':w,'seed':s,'method':m,'n':N,'task_pairs':N,'heldout_pair_nrmse':score(pred[m],tq,[4,5])[0],'nrmse_mean':err,'payload_bytes':len(b),'bytes_per_task_pair':len(b)/N,'logical_skills':4 if m!='no_view' else 2,'physical_modules':4 if m=='full' else 2,'skill_calibration_MAC':4*64*4,'view_search_MAC':2*361*64*4 if m.startswith('mirror') else 0,'query_MAC_per_example':4 if m!='mirror' else 8,'fit_wall_seconds':fitwall,'view_search_wall_seconds':viewwall if m.startswith('mirror') else 0.,'query_wall_seconds_mean':qwall,'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 if phase=='development':print(json.dumps({'phase':'development','runs':len(worlds)*3}));return
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run(a.parse_args().phase)
