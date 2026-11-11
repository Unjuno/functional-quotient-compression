import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=32; NROLE=4; NQUERY=64

def save(path,**a):
 np.savez(path,**a); n=path.stat().st_size
 with np.load(path) as z:
  assert set(z.files)==set(a)
  for k,v in a.items(): assert np.array_equal(z[k],v)
 path.unlink(); return n+len(json.dumps({'D':D,'roles':NROLE,'pairs':NROLE*NROLE}).encode())

def run(seed):
 rng=np.random.default_rng(seed)
 signs=rng.choice(np.array([-1.,1.],np.float32),size=(NROLE,D)); K=(signs/np.sqrt(D)).astype('<f4')
 V=rng.normal(size=(NROLE,D)).astype(np.float32);V/=np.linalg.norm(V,axis=1,keepdims=True);V=V.astype('<f4')
 pairs=[(i,j) for i in range(NROLE) for j in range(NROLE)]
 train=[(i,j) for i,j in pairs if (i+j)%2==0];held=[(i,j) for i,j in pairs if (i+j)%2==1]
 ops=np.stack([np.outer(V[j],K[i]) for i,j in pairs]).astype('<f4')
 bound=np.stack([signs[i]*V[j] for i,j in pairs]).astype('<f4')
 ids=np.array(pairs,dtype=np.uint8);path=ROOT/'source'/'payload.npz'
 sizes={'full':save(path,operators=ops,pair_ids=ids),'native':save(path,key_roles=K,value_roles=V,pair_ids=ids),'mirror':save(path,key_basis=K,value_basis=V,view_ids=ids),'hadamard':save(path,bound_pairs=bound,pair_ids=ids,key_signs=signs.astype('<f4'))}
 errs={k:[] for k in sizes}; macs={k:0 for k in sizes}; t0=time.perf_counter()
 for i,j in held:
  q=rng.normal(size=(NQUERY,D)).astype(np.float32)/np.sqrt(D)
  target=(q@K[i])[:,None]*V[j][None,:]
  # Independent full matrix pair.
  pred=q@ops[i*NROLE+j].T
  errs['full'].extend(np.mean((pred-target)**2,axis=1)/(np.mean(target**2,axis=1)+1e-12));macs['full']+=NQUERY*D*D
  # Native direct role-factor baseline and Mirror View pair composition.
  f=(q@K[i])[:,None]*V[j][None,:]
  for m in ('native','mirror'):
   errs[m].extend(np.mean((f-target)**2,axis=1)/(np.mean(target**2,axis=1)+1e-12));macs[m]+=NQUERY*2*D
  # MAP/Hadamard binding: unbind value with key signs then apply query-key similarity.
  decoded=bound[i*NROLE+j]*signs[i]
  h=(q@K[i])[:,None]*decoded[None,:]
  errs['hadamard'].extend(np.mean((h-target)**2,axis=1)/(np.mean(target**2,axis=1)+1e-12));macs['hadamard']+=NQUERY*2*D+D
 return dict(seed=seed,train_role_examples=len(train)*2,heldout_pairs=len(held),rows={m:dict(nrmse=float(np.sqrt(np.mean(errs[m]))),bytes=sizes[m],query_macs=macs[m]) for m in sizes},wall_time_s=time.perf_counter()-t0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();seeds=[31] if a.development else ([61,79] if a.fresh else [31,61,79]);print(json.dumps([run(s) for s in seeds],indent=2))
if __name__=='__main__':main()
