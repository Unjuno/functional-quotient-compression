#!/usr/bin/env python3
"""Synthetic Lennard-Jones NVE rollout screen, no MatterSim/MACE."""
import argparse,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
EPS=.01;SIG=1.;DT=.002;STEPS=3000
BASE=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)*.4

def ef(pos,eps=EPS,sigma=SIG,offset=None):
 e=0.;f=np.zeros_like(pos)
 for i in range(len(pos)):
  for j in range(i+1,len(pos)):
   d=pos[i]-pos[j];r=np.linalg.norm(d);sr6=(sigma/r)**6;sr12=sr6*sr6
   e+=4*eps*(sr12-sr6);v=24*eps*(2*sr12-sr6)/(r*r)*d;f[i]+=v;f[j]-=v
 if offset is not None:f+=offset
 return e,f

def rollout(seed,eps,offset=None):
 rng=np.random.default_rng(seed);x=BASE.copy();v=rng.normal(0,.003,x.shape);v-=v.mean(0);e0,f=ef(x,eps,offset=offset);h0=e0+.5*np.sum(v*v);maxdr=0.;trace=[]
 for k in range(STEPS):
  x=x+v*DT+.5*f*DT*DT;e,f2=ef(x,eps,offset=offset);v=v+.5*(f+f2)*DT;f=f2
  h=e+.5*np.sum(v*v);maxdr=max(maxdr,abs(h-h0)/(abs(h0)+1e-12))
  if k in [0,9,99,999,2999]:trace.append({'step':k+1,'energy':float(e),'total_energy':float(h)})
  if not np.all(np.isfinite(x)) or np.max(np.abs(x))>100:return {'stable':False,'max_relative_energy_drift':float('inf'),'trace':trace,'final_positions':x,'final_velocities':v}
 return {'stable':True,'max_relative_energy_drift':maxdr,'trace':trace,'final_positions':x,'final_velocities':v}

def run(seed):
 codes=np.array([0.,.15]);epses=EPS*(1+codes);outs={}
 for name,es in [('mirror',epses),('native',epses),('independent',epses)]:
  outs[name]=[rollout(seed+q,float(e)) for q,e in enumerate(es)]
 diff=max(float(np.max(np.abs(outs['mirror'][q]['final_positions']-outs['native'][q]['final_positions']))) for q in range(2))
 bad=rollout(seed,EPS*(1+codes[1]),offset=np.tile([1e-3,-2e-3,5e-4],(4,1)))
 states={'mirror':{'epsilon_base':np.array(EPS),'sigma':np.array(SIG),'codes':codes},'native':{'epsilon_per_task':epses,'sigma':np.array(SIG)},'independent':{'epsilon_per_task':epses,'sigma':np.array(SIG)}}
 sizes={};rt={}
 for n,st in states.items():
  p=RUNS/f'{seed}_{n}.npz';np.savez(p,**st);sizes[n]=p.stat().st_size;z=np.load(p);rt[n]=all(np.array_equal(z[k],v) for k,v in st.items())
 return {'world':seed,'steps':STEPS,'dt':DT,'methods':{n:{'serialized_bytes':sizes[n],'roundtrip_exact':rt[n],'trajectories':[{'stable':o['stable'],'max_relative_energy_drift':o['max_relative_energy_drift'],'trace':o['trace']} for o in outs[n]],'inference_mac_proxy_per_step':6*24+(2 if n=='mirror' else 0)} for n in states},'mirror_native_final_position_max_abs':diff,'negative_force_control':{'stable':bad['stable'],'max_relative_energy_drift':bad['max_relative_energy_drift'],'final_com_displacement':float(np.linalg.norm(bad['final_positions'].mean(0)-BASE.mean(0))),'trace':bad['trace']}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['dev','fresh','verify']);ap.add_argument('--world',type=int);a=ap.parse_args()
 if a.mode=='dev':
  r=run(100101);(RUNS/'development.json').write_text(json.dumps({'world':100101,'steps':STEPS,'dt':DT,'max_drift':max(x['max_relative_energy_drift'] for x in r['methods']['mirror']['trajectories'])},indent=2)+'\n');print(json.dumps(r,indent=2));return
 if a.mode=='fresh':
  if a.world not in [100111,100113,100117]:raise SystemExit('unregistered fresh world')
  t=time.perf_counter();r=run(a.world);r['fresh_batch_wall_time_s']=time.perf_counter()-t;(RUNS/f'fresh_{a.world}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return
 files=sorted(RUNS.glob('fresh_*.json'));checks=[]
 for p in files:
  d=json.load(open(p));r=run(d['world']);checks.append({'world':d['world'],'replay_drift_difference':max(abs(r['methods']['mirror']['trajectories'][i]['max_relative_energy_drift']-d['methods']['mirror']['trajectories'][i]['max_relative_energy_drift']) for i in range(2)),'mirror_native_position_diff':r['mirror_native_final_position_max_abs'],'stable':all(t['stable'] for t in r['methods']['mirror']['trajectories']),'roundtrip':all(x['roundtrip_exact'] for x in d['methods'].values())})
 print(json.dumps({'checks':checks,'all_pass':bool(checks) and all(x['replay_drift_difference']<1e-14 and x['mirror_native_position_diff']<1e-10 and x['stable'] and x['roundtrip'] for x in checks)},indent=2))
if __name__=='__main__':main()
