import argparse,csv,json,hashlib,time,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
D=8; TRAIN=[1,4]; HELD=[2,3,5,6]; ALL=[1,2,3,4,5,6]
def theta(t): return .12-.035*t
def giv(h,a):
 z=h.copy(); x=z[:,0].copy();y=z[:,1].copy();c=np.cos(a);s=np.sin(a);z[:,0]=c*x-s*y;z[:,1]=s*x+c*y;return z
def recurrent(x,A,W,t):
 h=np.tanh(x@A)
 for _ in range(t):h=np.tanh(h@W)
 return h
def run(seed,split,root):
 rng=np.random.default_rng(seed);A=rng.normal(0,.3,(D,D));W=np.eye(D)+rng.normal(0,.07,(D,D));C=rng.normal(0,.28,(D,D));X={t:rng.normal(size=(256,D)) for t in ALL}
 Y={t:giv(recurrent(X[t],A,W,t),theta(t))@C for t in ALL}
 def mirror(x,t,q):return giv(recurrent(x,A,W,t),q[0]+q[1]*t)@C
 def res(q):return np.concatenate([(mirror(X[t],t,q)-Y[t]).ravel() for t in TRAIN])
 st=time.perf_counter();q=least_squares(res,[0.,0.],max_nfev=50).x;fit=time.perf_counter()-st
 direct=q.copy();static={t:np.array([[np.cos(theta(t))-1,np.sin(theta(t))],[-np.sin(theta(t)),np.cos(theta(t))-1]]) for t in ALL};ind={t:np.eye(D) for t in ALL}
 for t in ALL:ind[t][:2,:2]=np.array([[np.cos(theta(t)),np.sin(theta(t))],[-np.sin(theta(t)),np.cos(theta(t))]])
 ind={t:ind[t]@C for t in ALL}
 out=Path(root)/split/str(seed);out.mkdir(parents=True,exist_ok=True)
 def save(name,**kw):p=out/f'{name}.npz';np.savez_compressed(p,**kw);return p.stat().st_size
 sb=save('shared',A=A,W=W,C=C);mb=save('mirror',q=q);db=save('direct_time',q=direct);rb=save('static_lora',Q=np.stack([static[t] for t in ALL]));ib=save('independent',**{f'M{t}':ind[t] for t in ALL})
 with np.load(out/'shared.npz') as z:A0,W0,C0=z['A'],z['W'],z['C']
 with np.load(out/'mirror.npz') as z:qm=z['q']
 with np.load(out/'direct_time.npz') as z:qd=z['q']
 with np.load(out/'static_lora.npz') as z:Qs=z['Q']
 with np.load(out/'independent.npz') as z:Mi={t:z[f'M{t}'] for t in ALL}
 methods={}
 for t in HELD:
  h=recurrent(X[t],A0,W0,t)
  methods[t]={'native':h@C0,'mirror':giv(h,qm[0]+qm[1]*t)@C0,'direct_time':giv(h,qd[0]+qd[1]*t)@C0,'static_lora':np.column_stack((h[:,0:2]@(np.eye(2)+Qs[t-1]),h[:,2:]))@C0,'independent':h@Mi[t]}
 sizes={'native':sb,'mirror':sb+mb,'direct_time':sb+db,'static_lora':sb+rb,'independent':sb+ib}
 rows=[];times={};macs={}
 for m in ['native','mirror','direct_time','static_lora','independent']:
  errs=[];ms=[];ops=[]
  for t in HELD:
   pred=methods[t][m];target=Y[t];errs.append(np.sqrt(np.mean((pred-target)**2)/(np.mean(target*target)+1e-20)))
   ops.append(256*D*D*(2+t)+ (256*4 if m in ('mirror','direct_time','static_lora') else 0))
  # Full forward includes recurrent transitions and output projection; averaged over heldout step counts.
  def fn():
   acc=[]
   for tt in HELD:
    h=recurrent(X[tt],A0,W0,tt)
    if m=='mirror':z=giv(h,qm[0]+qm[1]*tt)@C0
    elif m=='direct_time':z=giv(h,qd[0]+qd[1]*tt)@C0
    elif m=='static_lora':z=np.column_stack((h[:,0:2]@(np.eye(2)+Qs[tt-1]),h[:,2:]))@C0
    elif m=='independent':z=h@Mi[tt]
    else:z=h@C0
    acc.append(z)
   return acc
  for _ in range(2):fn()
  st=time.perf_counter()
  for _ in range(20):fn()
  times[m]=(time.perf_counter()-st)*1000/20;macs[m]=int(np.mean(ops))
  for t,e in zip(HELD,errs):rows.append({'seed':seed,'split':split,'step':t,'method':m,'nrmse':float(e),'payload_bytes':sizes[m],'active_macs':macs[m],'forward_ms_all_heldout':times[m]})
 hashes={k:hashlib.sha256((out/f'{k}.npz').read_bytes()).hexdigest() for k in ['shared','mirror','direct_time','static_lora','independent']}; objbytes={k:(out/f'{k}.npz').stat().st_size for k in hashes}
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'object_bytes':objbytes,'total_payload_bytes':sizes,'hashes':hashes,'mirror_q':qm.tolist(),'direct_q':qd.tolist(),'fit_seconds':fit,'active_macs':macs,'forward_ms':times},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();r=[]
 for s in a.seeds:r+=run(s,a.split,a.out)
 p=Path(a.out)/a.split/'RESULTS_CORE.csv';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
if __name__=='__main__':main()
