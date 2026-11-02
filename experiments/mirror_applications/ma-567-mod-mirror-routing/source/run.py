import argparse,csv,json,hashlib,time,math
from pathlib import Path
import numpy as np
D=8; PHI=.31; N=512; K=256

def role_angles(h,role,angles,role_codes):
 z=h.copy(); idx=np.searchsorted(role_codes,role); t=angles[idx]; c=np.cos(t); s=np.sin(t)
 a=h[:,0].copy(); b=h[:,1].copy(); z[:,0]=c*a-s*b; z[:,1]=s*a+c*b; return z

def route_mask(x,router):
 score=router['coeff'][0]*x[:,0]+router['coeff'][1]*x[:,1]; k=int(round(len(x)*router['fraction']))
 if router['kind']!=1: raise ValueError('unsupported serialized router kind')
 idx=np.argsort(score,kind='stable')[-k:]; mask=np.zeros(len(x),dtype=bool); mask[idx]=True; return mask

def hidden_selected(x,A,B,router):
 h1=np.tanh(x@A); route=route_mask(x,router); h=h1.copy()
 h[route]=np.tanh(.88*h1[route]@B+.12*h1[route])
 return h,route

def forward(x,role,weights,router,method,view):
 A,B,C=weights
 if method=='dense_upper':
  h1=np.tanh(x@A); h2=np.tanh(.88*h1@B+.12*h1); route=route_mask(x,router); h=h1.copy(); h[route]=h2[route]
 else: h,route=hidden_selected(x,A,B,router)
 if method in ('mod_mirror','mod_direct_angle','dense_upper'): y=role_angles(h,role,view['angles'],view['role_codes'])@C
 elif method=='mod_rank2':
  ids=np.searchsorted(view['role_codes'],role); q=view['coeff'][ids]; axes=view['basis_indices']; z=h.copy(); z[:,axes]=np.einsum('ni,nij->nj',h[:,axes],np.eye(2)[None,:,:]+q); y=z@C
 else: y=h@C
 return y,route

def run(seed,split,root):
 rng=np.random.default_rng(seed); A=rng.normal(0,.32,(D,D)); B=np.eye(D)+rng.normal(0,.08,(D,D)); C=rng.normal(0,.3,(D,D)); X=rng.normal(size=(N,D)); role=np.tile(np.array([-1.,1.]),N//2); rng.shuffle(role)
 out=Path(root)/split/str(seed); out.mkdir(parents=True,exist_ok=True)
 # Serialize first, then load the exact inference payload used by every prediction.
 def save(name,**kw):
  p=out/f'{name}.npz'; np.savez_compressed(p,**kw); return p.stat().st_size
 shared_b=save('shared',A=A,B=B,C=C,router_kind=np.array([1],np.int8),deep_fraction=np.array([.5]),difficulty_coeff=np.array([1.,.05]))
 angles=np.array([-PHI,PHI],np.float32)
 coeff=np.stack([np.array([[np.cos(-PHI)-1,np.sin(-PHI)],[-np.sin(-PHI),np.cos(-PHI)-1]],np.float32),np.array([[np.cos(PHI)-1,np.sin(PHI)],[-np.sin(PHI),np.cos(PHI)-1]],np.float32)])
 mirror_b=save('mirror',angles=angles,role_codes=np.array([-1,1],np.int8))
 direct_b=save('direct_angle',angles=angles,role_codes=np.array([-1,1],np.int8))
 rank2_b=save('rank2',coeff=coeff,basis_indices=np.array([0,1],np.int8),role_codes=np.array([-1,1],np.int8))
 dense_b=save('dense',angles=angles,role_codes=np.array([-1,1],np.int8),dense_compute=np.array([1],np.int8))
 with np.load(out/'shared.npz') as q: weights=(q['A'],q['B'],q['C']); router={'coeff':q['difficulty_coeff'],'fraction':float(q['deep_fraction'][0]),'kind':int(q['router_kind'][0])}
 with np.load(out/'mirror.npz') as q: mv={'angles':q['angles'],'role_codes':q['role_codes']}
 with np.load(out/'direct_angle.npz') as q: av={'angles':q['angles'],'role_codes':q['role_codes']}
 with np.load(out/'rank2.npz') as q: rv={'coeff':q['coeff'],'role_codes':q['role_codes'],'basis_indices':q['basis_indices']}
 with np.load(out/'dense.npz') as q: dv={'angles':q['angles'],'role_codes':q['role_codes']}
 methods={'mod_only':{},'mod_mirror':mv,'mod_direct_angle':av,'mod_rank2':rv,'dense_upper':dv}
 preds={}; routes={}
 for m,v in methods.items(): preds[m],routes[m]=forward(X,role,weights,router,m,v)
 # Independent teacher target uses the frozen high-precision phase, not decoded candidate state.
 h_teacher,_=hidden_selected(X,*weights[:2],router)
 ang_teacher=(role.astype(np.float64)*.31)
 ht=h_teacher.astype(np.float64).copy(); aa=ht[:,0].copy(); bb=ht[:,1].copy(); cc=np.cos(ang_teacher); ss=np.sin(ang_teacher); ht[:,0]=cc*aa-ss*bb; ht[:,1]=ss*aa+cc*bb
 target=ht@weights[2].astype(np.float64)
 # Cost ledger includes block 1, output projection, routed block 2, view/residual; router sorting is separately reported.
 active_k=int(round(N*router['fraction'])); base=N*D*D + N*D*D + active_k*D*D
 costs={'mod_only':base,'mod_mirror':base+N*4,'mod_direct_angle':base+N*4,'mod_rank2':base+N*4,'dense_upper':N*D*D+N*D*D+N*D*D+N*4}
 totals={'mod_only':shared_b,'mod_mirror':shared_b+mirror_b,'mod_direct_angle':shared_b+direct_b,'mod_rank2':shared_b+rank2_b,'dense_upper':shared_b+dense_b}
 # End-to-end CPU forward includes route score/ranking, block computation, role view/residual and output projection.
 funcs={m:(lambda m=m,v=v:forward(X,role,weights,router,m,v)[0]) for m,v in methods.items()}
 times={}
 for m,fn in funcs.items():
  for _ in range(5): fn()
  t=time.perf_counter()
  for _ in range(40): fn()
  times[m]=(time.perf_counter()-t)*1000/40
 rows=[]
 for m,y in preds.items():
  err=float(np.sqrt(np.mean((y-target)**2)/(np.mean(target*target)+1e-20)))
  rows.append({'seed':seed,'split':split,'method':m,'nrmse':err,'payload_bytes':totals[m],'active_macs':costs[m],'router_sort_ops_proxy':N*math.ceil(math.log2(N)),'selected_deep_fraction':float(routes[m].mean()),'executed_deep_fraction':1.0 if m=='dense_upper' else float(routes[m].mean()),'forward_ms_512':times[m]})
 filebytes={k:(out/f'{k}.npz').stat().st_size for k in ['shared','mirror','direct_angle','rank2','dense']}; hashes={k:hashlib.sha256((out/f'{k}.npz').read_bytes()).hexdigest() for k in filebytes}
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'split':split,'object_bytes':filebytes,'total_payload_bytes':totals,'hashes':hashes,'active_macs':costs,'router_sort_ops_proxy':N*math.ceil(math.log2(N)),'selected_deep_fraction':float(routes['mod_mirror'].mean()),'executed_deep_fraction':{'mod_only':float(routes['mod_only'].mean()),'mod_mirror':float(routes['mod_mirror'].mean()),'mod_direct_angle':float(routes['mod_direct_angle'].mean()),'mod_rank2':float(routes['mod_rank2'].mean()),'dense_upper':1.0},'forward_ms':times,'inference_dtype':'float64 shared backbone loaded from NPZ; float32 view tables loaded from NPZ'},indent=2)+'\n')
 with (out/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--split',choices=['dev','fresh'],required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();rows=[]
 for s in a.seeds:rows+=run(s,a.split,a.out)
 q=Path(a.out)/a.split/'RESULTS_CORE.csv';q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
