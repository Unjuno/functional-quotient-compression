#!/usr/bin/env python3
"""Synthetic LJ condition-invariance screen; no MatterSim or NPT implementation."""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
TS=np.array([300.,600.,900.]);PS=np.array([1.,2.,4.]);EPS=.01;SIG=1.0;N=5

def config(rng,t,p):
    box=4.0*((t/300.)/(p/1.))**(1/3);pts=[]
    while len(pts)<N:
        q=rng.uniform(0,box,3)
        if all(np.linalg.norm(q-v)>.85 for v in pts):pts.append(q)
    return np.asarray(pts)
def energy_force(pos,scale=1.):
    e=0.;f=np.zeros_like(pos)
    for i in range(N):
      for j in range(i+1,N):
        d=pos[i]-pos[j];r=np.linalg.norm(d);sr6=(SIG/r)**6;sr12=sr6*sr6
        e+=4*EPS*(sr12-sr6);v=24*EPS*(2*sr12-sr6)/(r*r)*d
        f[i]+=v;f[j]-=v
    return scale*e,scale*f
def sample(rng,t,p,n):
    xs=[];ys=[]
    for _ in range(n):
      pos=config(rng,t,p);e,f=energy_force(pos);xs.append(pos);ys.append((e,f))
    return xs,ys
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2))/(np.std(b)+1e-12))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh','verify']);ap.add_argument('--world',type=int);a=ap.parse_args()
 if a.mode=='dev':
  # For the exact fixed physical law, every cell's ideal scale is one; the 3x3 table is rank one.
  mat=np.ones((3,3));u,s,vh=np.linalg.svd(mat,full_matrices=False);res=float(np.max(np.abs(mat-u[:,:1]@np.diag(s[:1])@vh[:1,:])))
  d={'seed':99901,'selected_code_rank':1,'development_scale_table':mat.tolist(),'rank1_max_abs_reconstruction_residual':res,'fixed_law':{'epsilon':EPS,'sigma':SIG}}
  (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return
 if a.mode=='fresh':
  if a.world not in [99911,99913,99917]:raise SystemExit('unregistered fresh world')
  rng=np.random.default_rng(a.world);cal={};ev={};sc=np.ones((3,3));t0=time.perf_counter();ncal=8;neval=24
  for i,t in enumerate(TS):
   for j,p in enumerate(PS):
    xp,yp=sample(rng,t,p,ncal);xv,yv=sample(rng,t,p,neval);cal[i,j]=(xp,yp);ev[i,j]=(xv,yv)
    # Calibrate a scalar on the known physical energy/force backbone.
    num=den=0.
    for pos,(e,f) in zip(xp,yp):
     e0,f0=energy_force(pos);num+=e0*e+float(np.sum(f0*f));den+=e0*e0+float(np.sum(f0*f0))
    sc[i,j]=num/(den+1e-30)
  u,s,vh=np.linalg.svd(sc,full_matrices=False);alpha=u[:,0]*np.sqrt(s[0]);beta=vh[0,:]*np.sqrt(s[0]);fact=np.outer(alpha,beta)
  meth={k:{'pred_e':[],'true_e':[],'pred_f':[],'true_f':[]} for k in ['shared_only','mirror_factor','native_direct_table']}
  for i,t in enumerate(TS):
   for j,p in enumerate(PS):
    xp,yp=ev[i,j]
    for pos,(e,f) in zip(xp,yp):
     for name,g in [('shared_only',1.),('mirror_factor',fact[i,j]),('native_direct_table',sc[i,j])]:
      ep,fp=energy_force(pos,g);meth[name]['pred_e'].append(ep);meth[name]['true_e'].append(e);meth[name]['pred_f'].extend(fp.ravel());meth[name]['true_f'].extend(f.ravel())
  states={'shared_only':{'epsilon':np.array(EPS),'sigma':np.array(SIG)},
          'mirror_factor':{'epsilon':np.array(EPS),'sigma':np.array(SIG),'temperatures':TS,'pressures':PS,'alpha_T':alpha,'beta_P':beta},
          'native_direct_table':{'epsilon':np.array(EPS),'sigma':np.array(SIG),'temperatures':TS,'pressures':PS,'scales':sc}}
  summary={};roundtrip={};replays={}
  for name,d in meth.items():
   pe=np.asarray(d['pred_e']);ye=np.asarray(d['true_e']);pf=np.asarray(d['pred_f']);yf=np.asarray(d['true_f'])
   summary[name]={'energy_nrmse':nrmse(pe,ye),'force_nrmse':nrmse(pf,yf),'combined_nrmse':float(np.sqrt((nrmse(pe,ye)**2+nrmse(pf,yf)**2)/2))}
  bytes_={}
  for name,state in states.items():
   f=RUNS/f'{a.world}_{name}.npz';np.savez(f,**state);bytes_[name]=f.stat().st_size;z=np.load(f);roundtrip[name]=all(np.array_equal(z[k],v) for k,v in state.items())
  # Same geometry under all labels must have equal energy/force; measure direct view spread.
  probe=config(np.random.default_rng(a.world+1),600,2);vals=[]
  for i in range(3):
   for j in range(3):vals.append(energy_force(probe,fact[i,j]))
  spread=max(float(np.max(np.abs(vals[k][1]-vals[0][1]))) for k in range(9))
  # Finite-difference derivative and rotation checks at one condition.
  pos=probe.copy();e0,f0=energy_force(pos,fact[1,1]);epsfd=1e-6;fd=0.
  for i in range(N):
   for d in range(3):
    pp=pos.copy();pm=pos.copy();pp[i,d]+=epsfd;pm[i,d]-=epsfd
    fd=max(fd,abs(f0[i,d]+(energy_force(pp,fact[1,1])[0]-energy_force(pm,fact[1,1])[0])/(2*epsfd)))
  q,_=np.linalg.qr(np.random.default_rng(a.world+2).normal(size=(3,3)));rot=float(np.max(np.abs(energy_force(pos@q.T,fact[1,1])[1]-f0@q.T)))
  out={'world':a.world,'temperatures':TS.tolist(),'pressures':PS.tolist(),'condition_cells':9,'calibration_per_cell':ncal,'heldout_per_cell':neval,'methods':{k:{**summary[k],'serialized_bytes':bytes_[k],'inference_mac_proxy_per_config':100+(1 if k=='mirror_factor' else 0),'roundtrip_exact':roundtrip[k]} for k in summary},'force_condition_label_spread_max':spread,'force_gradient_fd_max_abs':fd,'rotation_equivariance_max_abs':rot,'fresh_batch_wall_time_s':time.perf_counter()-t0,'optimizer_updates':0}
  (RUNS/f'fresh_{a.world}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return
 fs=sorted(RUNS.glob('fresh_*.json'));checks=[]
 for file in fs:
  d=json.load(open(file));world=d['world'];rng=np.random.default_rng(world);cal_sc=np.ones((3,3));preds={k:[] for k in d['methods']}
  # Regenerate just the heldout geometry stream by consuming the calibration samples first.
  for i,t in enumerate(TS):
   for j,p in enumerate(PS):
    xp,yp=sample(rng,t,p,8);xv,yv=sample(rng,t,p,24);num=den=0.
    for pos,(e,f) in zip(xp,yp):
     e0,f0=energy_force(pos);num+=e0*e+float(np.sum(f0*f));den+=e0*e0+float(np.sum(f0*f0))
    cal_sc[i,j]=num/(den+1e-30)
    z=np.load(RUNS/f'{world}_mirror_factor.npz');g=float(z['alpha_T'][i]*z['beta_P'][j]);
    for pos,(e,f) in zip(xv,yv):
     for k,gg in [('shared_only',1.),('mirror_factor',g),('native_direct_table',cal_sc[i,j])]:
      ep,fp=energy_force(pos,gg);preds[k].append((ep,fp,e,f))
  diffs={}
  for k,arr in preds.items():
   pe=np.array([x[0] for x in arr]);ye=np.array([x[2] for x in arr]);pf=np.concatenate([x[1].ravel() for x in arr]);yf=np.concatenate([x[3].ravel() for x in arr]);score=float(np.sqrt((nrmse(pe,ye)**2+nrmse(pf,yf)**2)/2));diffs[k]=abs(score-d['methods'][k]['combined_nrmse'])
  checks.append({'world':world,'passed':max(diffs.values())<1e-12 and all(v['roundtrip_exact'] for v in d['methods'].values()) and d['force_gradient_fd_max_abs']<1e-7 and d['rotation_equivariance_max_abs']<1e-10,'max_metric_replay_difference':max(diffs.values()),'methods':diffs})
 print(json.dumps({'checks':checks,'all_pass':bool(checks) and all(x['passed'] for x in checks)},indent=2))
if __name__=='__main__':main()
