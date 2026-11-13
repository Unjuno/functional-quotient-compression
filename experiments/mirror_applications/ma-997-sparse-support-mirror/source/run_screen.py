#!/usr/bin/env python3
"""Synthetic shared-sparse-support conservative potential screen.
Radial design helpers are reused from the already versioned MA-996 toy harness.
"""
import sys,json,time,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'ma-996-mace-material-mirror'/'source'
sys.path.insert(0,str(PREV))
import run_screen as radial
D=radial.D; CENTERS=radial.CENTERS

def payload(path,**state): np.savez(path,**state); return path.stat().st_size

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh','verify']);ap.add_argument('--world',type=int);a=ap.parse_args()
 runs=ROOT/'runs';runs.mkdir(exist_ok=True)
 theta,_=radial.generator()
 if a.mode=='dev':
  rng=np.random.default_rng(99701); cs=[]
  for _ in range(12):
   delta=np.zeros(D);delta[[0,4,8,12,16,20]]=rng.normal(0,.03,6);c=theta+delta
   x,y=radial.dataset(rng,c,20,.65);cs.append(radial.fit(x,y))
  resid=np.stack(cs)-theta; support=np.argsort(np.mean(np.abs(resid),axis=0))[-6:];support=np.sort(support)
  d={'seed':99701,'support_indices':support.tolist(),'support_size':6,'dev_materials':12,'configurations_per_material':20}
  (runs/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
 elif a.mode=='fresh':
  if a.world not in [99711,99713,99717]:raise SystemExit('unregistered fresh world')
  cfg=json.load(open(runs/'development.json'));support=np.asarray(cfg['support_indices'],dtype=int);rng=np.random.default_rng(a.world);width=.65
  lists={k:[] for k in ['shared_only','mirror_shared_support','native_shared_support','independent_sparse','independent_full']}
  codes=[];inds=[];sparse_codes=[];fulls=[];checks=[];t0=time.perf_counter()
  for target in range(4):
   delta=np.zeros(D);delta[support]=rng.normal(0,.03,len(support));c=theta+delta
   xc,yc=radial.dataset(rng,c,8,width);xv,yv=radial.dataset(rng,c,24,width)
   cm=theta.copy();m=radial.fit(xc[:,support],yc-xc@theta);cm[support]+=m
   # Matched ordinary sparse/native mask uses the same support and fitted values.
   cn=theta.copy();cn[support]+=m.copy()
   # Independent sparse adaptation selects its own mask using calibration correlations.
   score=np.abs(xc.T@(yc-xc@theta));si=np.sort(np.argsort(score)[-6:]);mi=radial.fit(xc[:,si],yc-xc@theta);ci=theta.copy();ci[si]+=mi
   cf=radial.fit(xc,yc)
   for name,coef in [('shared_only',theta),('mirror_shared_support',cm),('native_shared_support',cn),('independent_sparse',ci),('independent_full',cf)]:
    lists[name].append(radial.measure(coef,xv,yv))
   codes.append(m);inds.append(si);sparse_codes.append(mi);fulls.append(cf)
   fd,eq=radial.force_consistency(cm,np.random.default_rng(a.world+target+5),width)
   checks.append({'target':target,'force_gradient_max_abs':fd,'rotation_equivariance_max_abs':eq,'calibration_examples':8,'heldout_examples':24})
  states={
   'shared_only':{'centers':CENTERS,'width':np.array(width),'base':theta},
   'mirror_shared_support':{'centers':CENTERS,'width':np.array(width),'base':theta,'indices':support.astype(np.int16),'values':np.stack(codes)},
   'native_shared_support':{'centers':CENTERS,'width':np.array(width),'base':theta,'indices':support.astype(np.int16),'values':np.stack(codes)},
   'independent_sparse':{'centers':CENTERS,'width':np.array(width),'base':theta,'indices':np.stack(inds).astype(np.int16),'values':np.stack(sparse_codes)},
   'independent_full':{'centers':CENTERS,'width':np.array(width),'coefficients':np.stack(fulls)}}
  sizes={};roundtrip={}
  for name,state in states.items():
   f=runs/f'{a.world}_{name}.npz';sizes[name]=payload(f,**state);z=np.load(f);roundtrip[name]=all(np.array_equal(z[k],v) for k,v in state.items())
  methods={}
  for name,vals in lists.items():
   arr=np.asarray(vals);methods[name]={'energy_nrmse_mean':float(arr[:,0].mean()),'force_nrmse_mean':float(arr[:,1].mean()),'combined_nrmse_mean':float(arr[:,2].mean()),'serialized_bytes':sizes[name],'inference_mac_proxy_per_config':100,'material_decode_mac_proxy':6 if name in ['mirror_shared_support','native_shared_support'] else 6 if name=='independent_sparse' else 0,'roundtrip_exact':roundtrip[name]}
  r={'world':a.world,'support_indices':support.tolist(),'targets':4,'calibration_examples_per_target':8,'heldout_examples_per_target':24,'methods':methods,'integrity_checks':checks,'optimizer_updates':0,'fresh_batch_wall_time_s':time.perf_counter()-t0}
  (runs/f'fresh_{a.world}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
 else:
  fs=sorted(runs.glob('fresh_*.json'));ok=[]
  for f in fs:
   d=json.load(open(f));ok.append(all(x['roundtrip_exact'] for x in d['methods'].values()) and max(x['force_gradient_max_abs'] for x in d['integrity_checks'])<1e-7 and max(x['rotation_equivariance_max_abs'] for x in d['integrity_checks'])<1e-10)
  print(json.dumps({'files':[str(f) for f in fs],'passed':ok,'all_pass':bool(ok) and all(ok)},indent=2))
if __name__=='__main__':main()
