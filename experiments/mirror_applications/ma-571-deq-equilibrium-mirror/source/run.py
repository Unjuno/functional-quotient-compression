import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
D=8;I=4;N=128;PHI=.23;TOL=1e-7;MAXIT=100;LAMBDA=.2

def rot_apply(u,theta):
 z=u.copy(); a=u[...,0].copy();b=u[...,1].copy();c=np.cos(theta);s=np.sin(theta);z[...,0]=c*a-s*b;z[...,1]=s*a+c*b;return z

def solve(x,U,A,V,angles=None,roles=None,Q=None,indU=None):
 if indU is not None: u=np.stack([x[k]@indU[int(roles[k]>0)] for k in range(len(x))])
 else:
  u=x@U
  if angles is not None:u=rot_apply(u,angles[np.searchsorted(np.array([-1.,1.]),roles)])
  elif Q is not None:
   q=Q[np.searchsorted(np.array([-1.,1.]),roles)];u0=u.copy();u[:,:2]=np.einsum('ni,nij->nj',u[:,:2],np.eye(2)[None,:,:]+q)
 z=np.zeros((len(x),D));iters=np.zeros(len(x),int);res=np.full(len(x),np.inf)
 for k in range(1,MAXIT+1):
  zn=np.tanh(LAMBDA*(z@A)+u);delta=np.max(np.abs(zn-z),axis=1);active=iters==0;done=(delta<TOL)&active;iters[done]=k;res[done]=delta[done];z[active]=zn[active]
  if np.all(iters):break
 iters[iters==0]=MAXIT;res[iters==MAXIT]=np.max(np.abs(np.tanh(LAMBDA*(z@A)+u)-z),axis=1)[iters==MAXIT]
 return z@V,iters,res

def run(seed,split,root):
 rng=np.random.default_rng(seed);U=rng.normal(0,.28,(I,D));A=rng.normal(0,.12,(D,D));V=rng.normal(0,.3,(D,I));X=rng.normal(size=(N,I));roles=np.tile(np.array([-1.,1.]),N//2);rng.shuffle(roles)
 out=Path(root)/split/str(seed);out.mkdir(parents=True,exist_ok=True)
 def save(k,**kw):p=out/f'{k}.npz';np.savez_compressed(p,**kw);return p.stat().st_size
 sb=save('shared',U=U,A=A,V=V,lambda_=np.array([LAMBDA]),tolerance=np.array([TOL]),max_iterations=np.array([MAXIT]),role_codes=np.array([-1,1],np.int8))
 angles=np.array([-PHI,PHI],np.float32);mb=save('mirror',angles=angles,role_codes=np.array([-1,1],np.int8));db=save('direct',angles=angles,role_codes=np.array([-1,1],np.int8))
 qs=np.stack([np.array([[np.cos(PHI*r)-1,np.sin(PHI*r)],[-np.sin(PHI*r),np.cos(PHI*r)-1]]) for r in [-1.,1.]]);qb=save('rank2',Q=qs,basis_indices=np.array([0,1],np.int8),role_codes=np.array([-1,1],np.int8))
 indU=np.stack([rot_apply(U,PHI*r) for r in [-1.,1.]]);ib=save('independent',U_roles=indU)
 # Load every paid inference object before decoding.
 with np.load(out/'shared.npz') as q:U0,A0,V0=q['U'],q['A'],q['V'];lam=float(q['lambda_'][0]);tol=float(q['tolerance'][0]);maxit=int(q['max_iterations'][0]);roles0=q['role_codes']
 with np.load(out/'mirror.npz') as q:mv=(q['angles'],q['role_codes'])
 with np.load(out/'direct.npz') as q:dv=(q['angles'],q['role_codes'])
 with np.load(out/'rank2.npz') as q:rv=(q['Q'],q['role_codes'],q['basis_indices'])
 with np.load(out/'independent.npz') as q:iu=q['U_roles']
 target,_,_=solve(X,U0,A0,V0,angles=np.array([-PHI,PHI]),roles=roles)
 funcs={'native':lambda:solve(X,U0,A0,V0)[0],'mirror':lambda:solve(X,U0,A0,V0,angles=mv[0],roles=roles)[0],'direct':lambda:solve(X,U0,A0,V0,angles=dv[0],roles=roles)[0],'rank2':lambda:solve(X,U0,A0,V0,roles=roles,Q=rv[0])[0],'independent':lambda:solve(X,U0,A0,V0,roles=roles,indU=iu)[0]}
 payload={'native':sb,'mirror':sb+mb,'direct':sb+db,'rank2':sb+qb,'independent':sb+ib};obj={k:(out/f'{k}.npz').stat().st_size for k in ['shared','mirror','direct','rank2','independent']};hashes={k:hashlib.sha256((out/f'{k}.npz').read_bytes()).hexdigest() for k in obj}
 times={};rows=[]
 for m,fn in funcs.items():
  y,its,res=solve(X,U0,A0,V0,angles=mv[0],roles=roles) if m=='mirror' else solve(X,U0,A0,V0,angles=dv[0],roles=roles) if m=='direct' else solve(X,U0,A0,V0,roles=roles,Q=rv[0]) if m=='rank2' else solve(X,U0,A0,V0,roles=roles,indU=iu) if m=='independent' else solve(X,U0,A0,V0)
  nrmse=float(np.sqrt(np.mean((y-target)**2)/(np.mean(target*target)+1e-20)))
  for _ in range(2):fn()
  t=time.perf_counter()
  for _ in range(12):fn()
  times[m]=(time.perf_counter()-t)*1000/12
  avg_it=float(np.mean(its));macs=int(N*I*D+np.sum(its)*D*D+N*D*I+(N*4 if m!='native' else 0))
  rows.append({'seed':seed,'split':split,'method':m,'nrmse':nrmse,'max_fixed_point_residual':float(np.max(res)),'mean_iterations':avg_it,'payload_bytes':payload[m],'active_macs':macs,'forward_ms_128':times[m]})
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'object_bytes':obj,'total_payload_bytes':payload,'hashes':hashes,'mean_iterations':{r['method']:r['mean_iterations'] for r in rows},'max_residual':{r['method']:r['max_fixed_point_residual'] for r in rows},'active_macs':{r['method']:r['active_macs'] for r in rows},'forward_ms':times},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();r=[]
 for s in a.seeds:r+=run(s,a.split,a.out)
 q=Path(a.out)/a.split/'RESULTS_CORE.csv';q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
if __name__=='__main__':main()
