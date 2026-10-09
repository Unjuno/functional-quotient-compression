#!/usr/bin/env python3
import csv,hashlib,io,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts';PAY=ART/'payloads';DEV=[47000,47001];FRESH=[47010,47011,47012];SEEDS=[0,1,2];DIM=8
def rot(theta,i,j):
 R=torch.eye(DIM);c,s=torch.cos(torch.tensor(theta)),torch.sin(torch.tensor(theta));R[i,i]=c;R[i,j]=-s;R[j,i]=s;R[j,j]=c;return R
def atomdict(w,s):
 g=torch.Generator().manual_seed(w*1000003+s*997+53);u=torch.randn(DIM,generator=g);u=u/u.norm();v=torch.randn(DIM,generator=torch.Generator().manual_seed(w*31337+s*199));v=v/v.norm()
 u1=rot(.8,0,1)@u;v1=rot(-.65,0,1)@v;u2=rot(.9,2,3)@u;v2=rot(-.7,2,3)@v
 u3=torch.randn(DIM,generator=torch.Generator().manual_seed(w*9973+s*71));u3=u3/u3.norm();v3=torch.randn(DIM,generator=torch.Generator().manual_seed(w*8009+s*37));v3=v3/v3.norm()
 U=torch.stack([u,u1,u2,u3]);V=torch.stack([v,v1,v2,v3]);return U,V,{'u':u,'v':v,'angles':torch.tensor([.8,-.65,.9,-.7]),'u_private':u3,'v_private':v3}
def editset(w,s):
 U,V,basis=atomdict(w,s);g=torch.Generator().manual_seed(w*99131+s*71+17);edits=[];coeff=[]
 for i in range(64):
  idx=torch.randperm(4,generator=g)[:2];c=torch.randn(2,generator=torch.Generator().manual_seed(w*1217+s*79+i*31));D=c[0]*torch.outer(U[idx[0]],V[idx[0]])+c[1]*torch.outer(U[idx[1]],V[idx[1]]);edits.append(D);coeff.append((idx,c))
 return U,V,basis,torch.stack(edits),coeff
def project(D,U,V):
 mats=torch.stack([torch.outer(U[j],V[j]).reshape(-1) for j in range(len(U))],1);y=D.reshape(-1);return torch.linalg.lstsq(mats,y).solution
def top2(c):
 ix=torch.topk(c.abs(),min(2,len(c))).indices;return ix,c[ix]
def reconstruct(U,V,idx,coef):
 z=torch.zeros(DIM,DIM)
 for i,j in enumerate(idx):z+=coef[i]*torch.outer(U[int(j)],V[int(j)])
 return z
def mend_factor(D):
 u,s,vh=torch.linalg.svd(D,full_matrices=False);return u[:,:2]*s[:2],vh[:2,:]
def package(method,basis,indices,coeffs,N,mend):
 obj={'format':'ma470-v1','method':method,'N':N}
 if method=='mend':obj['rank2_factors']=[mend[i] for i in range(N)]
 elif method=='generic_basis':obj.update({'atom_U':basis['U'],'atom_V':basis['V'],'atom_indices':torch.stack(indices[:N]),'coefficients':torch.stack(coeffs[:N])})
 elif method=='mirror':obj.update({'shared_u':basis['mirror']['u'],'shared_v':basis['mirror']['v'],'view_angles':basis['mirror']['angles'],'atom_indices':torch.stack(indices[:N]),'coefficients':torch.stack(coeffs[:N])})
 elif method=='mirror_private':obj.update({'shared_u':basis['mirror']['u'],'shared_v':basis['mirror']['v'],'view_angles':basis['mirror']['angles'],'private_u':basis['mirror']['u_private'],'private_v':basis['mirror']['v_private'],'atom_indices':torch.stack(indices[:N]),'coefficients':torch.stack(coeffs[:N])})
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def evaluate(Ds,recons,U,V):
 err=[];loc=[];wall=[]
 for d,r in zip(Ds,recons):
  err.append(float((r-d).norm()/(d.norm()+1e-9)))
  # inputs orthogonal to both active atom key vectors: the true edit has zero effect.
  q=torch.randn(64,DIM);v=torch.linalg.svd(d,full_matrices=False).Vh[:2,:];q=q-(q@v.T)@v;t=time.perf_counter();drift=q@r.T;wall.append(time.perf_counter()-t);loc.append(float(drift.norm()/(q.norm()*d.norm()+1e-9)))
 return statistics.mean(err),statistics.mean(loc),statistics.mean(wall)
def run(phase):
 worlds=DEV if phase=='development' else FRESH;rows=[]
 if phase=='development':
  for w in worlds:
   for s in range(3):editset(w,s)
  (ART/'development_runs.json').write_text(json.dumps({'worlds':worlds,'seeds':SEEDS,'settings':'fixed basis geometry and LS projection'},indent=2)+'\n');return
 for w in worlds:
  for s in SEEDS:
   U,V,mb,Ds,teacher=editset(w,s);basis={'U':U,'V':V,'mirror':mb};mends=[mend_factor(d) for d in Ds]
   genericI=[];genericC=[];mirrorI=[];mirrorC=[];privI=[];privC=[];singleI=[];singleC=[]
   for d in Ds:
    cg=project(d,U,V);ig,vg=top2(cg);genericI.append(ig.to(torch.uint8));genericC.append(vg)
    U3=torch.stack([U[0],U[1],U[2]]);V3=torch.stack([V[0],V[1],V[2]]);cm=project(d,U3,V3);im,vm=top2(cm);mirrorI.append(im.to(torch.uint8));mirrorC.append(vm)
    ip,vp=top2(cg);privI.append(ip.to(torch.uint8));privC.append(vp)
    cs=project(d,U[:1],V[:1]);is_,vs=top2(cs);singleI.append(is_.to(torch.uint8));singleC.append(vs)
   recs={
    'mend':Ds,
    'generic_basis':[reconstruct(U,V,genericI[i],genericC[i]) for i in range(64)],
    'mirror':[reconstruct(U[:3],V[:3],mirrorI[i],mirrorC[i]) for i in range(64)],
    'mirror_private':[reconstruct(U,V,privI[i],privC[i]) for i in range(64)],
    'single':[reconstruct(U[:1],V[:1],singleI[i],singleC[i]) for i in range(64)]}
   for method in recs:
    for N in [1,20,64]:
     if method=='mend':inds=[];co=[]
     elif method=='generic_basis':inds,co=genericI,genericC
     elif method=='mirror':inds,co=mirrorI,mirrorC
     elif method=='mirror_private':inds,co=privI,privC
     else:inds,co=singleI,singleC
     bb=package(method,basis,inds,co,N,mends);path=PAY/f'{w}_{s}_{method}_N{N}.pt';path.write_bytes(bb)
     er,local,qwall=evaluate(Ds[:N],recs[method][:N],U,V)
     # Isolated encoding wall-clock for the amortized edit payload. This is
     # reported as a measurement, not used for protocol selection.
     t0=time.perf_counter()
     for _ in range(20): package(method,basis,inds,co,N,mends)
     edit_latency=(time.perf_counter()-t0)/20/N
     rows.append({'world':w,'seed':s,'method':method,'n':N,'edits':N,'edit_nrmse':er,'locality_drift':local,'payload_bytes':len(bb),'bytes_per_edit':len(bb)/N,'coefficient_projection_MAC_per_edit':256 if method!='mend' else 0,'update_apply_MAC_per_edit':128 if method!='mend' else 64,'edit_latency_seconds_per_edit':edit_latency,'query_wall_seconds_mean':qwall,'hash':hashlib.sha256(bb).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(ART/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':phase,'rows':len(rows)}))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['development','fresh'],required=True);run(a.parse_args().phase)
