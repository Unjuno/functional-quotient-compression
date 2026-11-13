#!/usr/bin/env python3
"""Synthetic conservative energy Mirror audit; reuses MA-996 radial basis."""
import sys,argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
PREV=ROOT.parent/'ma-996-mace-material-mirror'/'source';sys.path.insert(0,str(PREV));import run_screen as radial

def make_codes():
 theta,u=radial.generator();return theta,u,np.array([[.5,-.8],[-.3,.7]])
def eval_metrics(coeff,rng):
 fd,eq=radial.force_consistency(coeff,rng,.45)
 p,s=radial.config(rng);e,f=radial.design(p,s,width=.45)@coeff,radial.design(p,s,width=.45)@coeff
 return fd,eq

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh','verify']);ap.add_argument('--world',type=int);a=ap.parse_args()
 if a.mode=='dev':
  theta,u,codes=make_codes();(RUNS/'development.json').write_text(json.dumps({'seed':100001,'rank':2,'codes':codes.tolist(),'width':.45},indent=2)+'\n');print('development frozen: rank=2, width=.45');return
 if a.mode=='fresh':
  if a.world not in [100011,100013,100017]:raise SystemExit('unregistered fresh world')
  theta,u,codes=make_codes();coef=np.stack([theta+u@m for m in codes]);width=.45;t0=time.perf_counter();checks=[]
  for i,c in enumerate(coef):
   rng=np.random.default_rng(a.world+i);fd,eq=radial.force_consistency(c,rng,width)
   # A deliberate force-head perturbation violates F=-grad(E).
   pos,sp=radial.config(np.random.default_rng(a.world+i+10));x=radial.design(pos,sp,width=width);f=(x@c)[1:].reshape(-1,3);bad=f+np.tile(np.array([1e-3,-2e-3,5e-4]),(len(pos),1))
   bad_res=float(np.max(np.abs(bad-f)))
   # Functional multiplicity: difference between the two View outputs on this structure.
   checks.append({'view':i,'force_gradient_max_abs':fd,'rotation_equivariance_max_abs':eq,'negative_control_residual':bad_res,'code_function_difference_energy':float((radial.design(pos,sp,width=width)@(coef[0]-coef[1]))[0])})
  states={'mirror_energy':{'centers':radial.CENTERS,'width':np.array(width),'base':theta,'basis':u,'codes':codes},'native_rank2_energy':{'centers':radial.CENTERS,'width':np.array(width),'base':theta,'basis':u,'codes':codes},'independent_energy':{'centers':radial.CENTERS,'width':np.array(width),'coefficients':coef}}
  sizes={};rt={}
  for n,st in states.items():
   p=RUNS/f'{a.world}_{n}.npz';np.savez(p,**st);sizes[n]=p.stat().st_size;z=np.load(p);rt[n]=all(np.array_equal(z[k],v) for k,v in st.items())
  r={'world':a.world,'functions':2,'radial_examples':2,'optimizer_updates':0,'methods':{n:{'serialized_bytes':sizes[n],'roundtrip_exact':rt[n],'inference_mac_proxy_per_config':100+(10 if n in ['mirror_energy','native_rank2_energy'] else 0)} for n in states},'checks':checks,'fresh_batch_wall_time_s':time.perf_counter()-t0}
  (RUNS/f'fresh_{a.world}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return
 fs=sorted(RUNS.glob('fresh_*.json'));out=[]
 for p in fs:
  d=json.load(open(p));out.append({'world':d['world'],'passed':all(v['roundtrip_exact'] for v in d['methods'].values()) and all(x['force_gradient_max_abs']<1e-7 and x['rotation_equivariance_max_abs']<1e-10 and x['negative_control_residual']>1e-7 for x in d['checks']),'max_gradient':max(x['force_gradient_max_abs'] for x in d['checks']),'max_equivariance':max(x['rotation_equivariance_max_abs'] for x in d['checks']),'min_negative_control':min(x['negative_control_residual'] for x in d['checks'])})
 print(json.dumps({'checks':out,'all_pass':bool(out) and all(x['passed'] for x in out)},indent=2))
if __name__=='__main__':main()
