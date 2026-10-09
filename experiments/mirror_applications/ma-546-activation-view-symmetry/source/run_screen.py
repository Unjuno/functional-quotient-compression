#!/usr/bin/env python3
"""Screen exact hidden-coordinate gauge symmetry versus uncompensated views."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEV=(54601,54602); FRESH=(54611,54612,54613)
IN_D=32; H=64; OUT_D=32; N=256

def make(seed):
 rng=np.random.default_rng(seed)
 w1=rng.normal(0,.12,(H,IN_D)).astype(np.float32); w2=rng.normal(0,.12,(OUT_D,H)).astype(np.float32); x=rng.normal(0,1,(N,IN_D)).astype(np.float32)
 q,_=np.linalg.qr(rng.normal(size=(H,H))); q=q.astype(np.float32)
 perm=rng.permutation(H); p=np.eye(H,dtype=np.float32)[perm]; signs=rng.choice(np.array([-1,1],np.float32),size=H); ps=p*signs[None,:]
 scales=rng.uniform(.5,2,size=H).astype(np.float32); d=np.diag(scales)
 return w1,w2,x,{'permutation_sign':ps,'diagonal':d,'orthogonal':q}
def sha(b):return hashlib.sha256(b).hexdigest()
def write_payload(path,arrays):
 np.savez(path,**arrays); b=path.read_bytes(); return len(b),sha(b)
def run(seed,out):
 if seed in FRESH and not json.loads((ROOT/'runs/dev/DEV_GATE.json').read_text()).get('fresh_opened'):raise RuntimeError('fresh sealed by gate')
 out.mkdir(parents=True,exist_ok=True); w1,w2,x,views=make(seed); base=x+(x@w1.T)@w2.T; rows=[]; payloads={}; maxpaired=0.; mincond=float('inf'); maxunpaired=0.
 schema=np.array([546,IN_D,H,OUT_D,N],np.int32)
 for name,g in [('identity',np.eye(H,dtype=np.float32)),*views.items()]:
  inv=np.linalg.inv(g).astype(np.float32); cond=float(np.linalg.cond(g)); mincond=min(mincond,cond); a=g@w1; b=w2@inv
  paired=x+(x@a.T)@b.T; unpaired=x+(x@a.T)@w2.T
  pe=float(np.max(np.abs(paired-base))); rel=float(np.linalg.norm(paired-base)/max(np.linalg.norm(base),1e-12)); ue=float(np.linalg.norm(unpaired-base)/max(np.linalg.norm(base),1e-12)); maxpaired=max(maxpaired,pe);maxunpaired=max(maxunpaired,ue)
  t0=time.perf_counter(); common={'schema':schema,'seed':np.array([seed],np.int64),'view_name':np.array([name.encode()],dtype='S32'),'condition_number':np.array([cond],np.float64)}
  n,h=write_payload(out/f'{name}.npz',{**common,'W1_view':a,'W2_view':b,'G':g,'G_inverse':inv}); payloads[name]={'bytes':n,'sha256':h}
  write_s=time.perf_counter()-t0
  un,uh=write_payload(out/f'{name}_unpaired.npz',{**common,'W1_view':a,'W2_shared':w2,'G':g,'G_inverse':inv}); payloads[name+'_unpaired']={'bytes':un,'sha256':uh}
  rows.append({'view':name,'condition_number':cond,'paired_max_abs':pe,'paired_relative_rms':rel,'unpaired_relative_rms':ue,'paired_payload_bytes':n,'unpaired_payload_bytes':un,'write_seconds':write_s})
 baseline=write_payload(out/'baseline.npz',{'W1':w1,'W2':w2,'schema':schema,'seed':np.array([seed],np.int64)})[0]
 gate=maxpaired<=1e-5 and maxunpaired>1e-5 and mincond>=.25
 (out/'metrics.json').write_text(json.dumps({'seed':seed,'rows':rows,'max_paired_abs_error':maxpaired,'max_unpaired_relative_rms':maxunpaired,'min_condition_number':mincond,'baseline_payload_bytes':baseline,'payloads':payloads,'input_vectors':N,'optimizer_updates':0,'baseline_mac_per_input':OUT_D*H+H*IN_D},indent=2)+'\n')
 return gate,rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--condition',choices=['dev','fresh']);ap.add_argument('--out',type=Path,default=ROOT/'runs');args=ap.parse_args()
 if args.seed is not None: seeds=[args.seed];cond='fresh' if args.seed in FRESH else 'dev'
 else: cond=args.condition or 'dev';seeds=DEV if cond=='dev' else FRESH
 t=time.perf_counter(); results=[]
 for s in seeds:
  g,rows=run(s,args.out/cond/f'seed_{s}');results.append({'seed':s,'gate':g,'rows':rows})
 elapsed=time.perf_counter()-t; opened=all(r['gate'] for r in results) and cond=='dev'
 if cond=='dev':(args.out/'dev/DEV_GATE.json').write_text(json.dumps({'seeds':[r['seed'] for r in results],'checks':[{'seed':r['seed'],'paired_invariance':r['gate']} for r in results],'fresh_opened':opened},indent=2)+'\n')
 print(json.dumps({'condition':cond,'wall_seconds':elapsed,'fresh_opened':opened,'results':results},indent=2))
if __name__=='__main__':main()
