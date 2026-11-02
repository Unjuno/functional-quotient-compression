import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
D=8; PHI=.31

def rotate(h,role):
 z=h.copy(); a=z[:,0].copy(); b=z[:,1].copy(); ang=PHI*role; co=np.cos(ang); si=np.sin(ang); z[:,0]=co*a-si*b; z[:,1]=si*a+co*b; return z

def run(seed,split,root):
 rng=np.random.default_rng(seed); A=rng.normal(0,.32,(D,D)); B=np.eye(D)+rng.normal(0,.08,(D,D)); C=rng.normal(0,.3,(D,D)); X=rng.normal(size=(512,D)); role=np.tile(np.array([-1.,1.]),256); rng.shuffle(role)
 # MoD's fixed 50% compute route: route largest input difficulty scores to block 2.
 score=X[:,0]+.05*X[:,1]; route=np.zeros(512,dtype=bool); route[np.argsort(score)[-256:]]=True
 h1=np.tanh(X@A); h2=np.tanh(.88*h1@B+.12*h1); H=h1.copy(); H[route]=h2[route]
 Y=rotate(H,role)@C
 preds={}
 preds['mod_only']=H@C
 preds['mod_mirror']=rotate(H,role)@C
 # Generic rank-2 residual table. Each role stores a 2x2 correction; shared right factor C[:2] is charged once in shared state.
 coeff=np.stack([np.array([[np.cos(PHI*r)-1,np.sin(PHI*r)],[-np.sin(PHI*r),np.cos(PHI*r)-1]]) for r in [-1.,1.]])
 Q=coeff[(role>0).astype(int)]
 preds['mod_rank2']=H@C+np.einsum('ni,nij,jk->nk',H[:,:2,None].squeeze(-1),Q,C[:2,:])
 # Dense compute upper: execute block 2 for every token, then select the same routed function; quality equality, excess compute.
 DENSE=h2.copy(); DENSE[~route]=h1[~route]; preds['dense_upper']=rotate(DENSE,role)@C
 out=Path(root)/split/str(seed); out.mkdir(parents=True,exist_ok=True)
 def save(name,**kw):
  p=out/f'{name}.npz'; np.savez_compressed(p,**kw); return p.stat().st_size
 common=save('shared',A=A,B=B,C=C,route_threshold=np.array([0.]),route_rule=np.array([1],dtype=np.int8)); mb=save('mirror',angles=np.array([-PHI,PHI],dtype=np.float32),role_codes=np.array([-1,1],dtype=np.int8)); rb=save('rank2',coeff=coeff.astype(np.float32),basis_indices=np.array([0,1],dtype=np.int8)); db=save('dense',dense_flag=np.array([1],dtype=np.int8))
 sizes={'mod_only':common,'mod_mirror':common+mb,'mod_rank2':common+rb,'dense_upper':common+db}
 costs={'mod_only':512*D*D+256*D*D,'mod_mirror':512*D*D+256*D*D+512*4,'mod_rank2':512*D*D+256*D*D+512*(4*2+2*D),'dense_upper':512*D*D+512*D*D+512*4}
 times={}
 for name,fn in {'mod_only':lambda:H@C,'mod_mirror':lambda:rotate(H,role)@C,'mod_rank2':lambda:H@C+np.einsum('ni,nij,jk->nk',H[:,:2],Q,C[:2,:]),'dense_upper':lambda:rotate(DENSE,role)@C}.items():
  for _ in range(3):fn()
  t=time.perf_counter()
  for _ in range(40):fn()
  times[name]=(time.perf_counter()-t)*1000/40
 rows=[]
 for name,y in preds.items():
  err=np.sqrt(np.mean((y-Y)**2)/(np.mean(Y*Y)+1e-20)); rows.append({'seed':seed,'split':split,'method':name,'nrmse':err,'payload_bytes':sizes[name],'active_macs':costs[name],'deep_token_fraction':float(route.mean()),'forward_ms_512':times[name]})
 files={x:out/f'{x}.npz' for x in ['shared','mirror','rank2','dense']}; hashes={x:hashlib.sha256(f.read_bytes()).hexdigest() for x,f in files.items()}
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'bytes':sizes,'hashes':hashes,'active_macs':costs,'deep_token_fraction':float(route.mean()),'times_ms':times},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();rows=[]
 for s in a.seeds:rows+=run(s,a.split,a.out)
 q=Path(a.out)/a.split/'RESULTS_CORE.csv';q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
