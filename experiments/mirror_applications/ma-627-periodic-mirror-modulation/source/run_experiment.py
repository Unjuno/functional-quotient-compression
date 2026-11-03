import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];PHASES=np.array([0,np.pi/2,np.pi,3*np.pi/2]);HARM=np.arange(1,5);TASKS=[(k,p) for k in HARM for p in range(4)];HELD={(1,0),(2,1),(3,2),(4,3)}
def f(x,a,k,phi):return a*np.sin(2*np.pi*k*x+phi)
def nrmse(y,z):return float(np.sqrt(np.mean((y-z)**2))/max(np.sqrt(np.mean(y*y)),1e-12))
def interp(v,x):
 u=np.clip(x*(len(v)-1),0,len(v)-1-1e-9);i=np.floor(u).astype(int);a=u-i;return (1-a)*v[i]+a*v[i+1]
def run(seed):
 rng=np.random.default_rng(seed);amps={t:rng.uniform(.5,1.5) for t in TASKS};xs={};xa={};codes={};curves={};fitwork={}
 for task in TASKS:
  k,p=task;x=rng.uniform(0,1,64);z=rng.uniform(0,1,512);y=f(x,amps[task],k,PHASES[p]);best=None
  for kh in HARM:
   mat=np.stack([np.sin(2*np.pi*kh*x),np.cos(2*np.pi*kh*x)],axis=1);coef=np.linalg.lstsq(mat,y,rcond=None)[0];err=np.mean((mat@coef-y)**2)
   if best is None or err<best[0]:best=(err,kh,coef)
  _,kh,c=best;aa=float(np.linalg.norm(c));phi=float(np.arctan2(c[1],c[0]));codes[task]=[aa,float(kh),phi];xs[task]=x;xa[task]=z
  gridx=np.linspace(0,1,512);curves[task]=f(gridx,amps[task],k,PHASES[p]).astype(np.float32);fitwork[task]=4*64*4+2*64*2
 codearr=np.array([codes[t] for t in TASKS]);ids=np.array(TASKS,dtype=np.int8);schema=np.frombuffer(b'PERIODIC-SIN-V1',dtype=np.uint8)
 payload={'shared_mirror_decoder':{'codes':codearr,'task_indices':ids,'decoder_schema':schema},'native_periodic_activation':{'codes':codearr,'task_indices':ids,'decoder_schema':schema},'independent_sampled_curves':{'curves':np.stack([curves[t] for t in TASKS]),'task_indices':ids,'curve_schema':np.frombuffer(b'PERIODIC-GRID512-LINEAR',dtype=np.uint8)}}
 sizes={}
 for method,arr in payload.items():
  p=R/'source'/f'{seed}_{method}.npz';np.savez_compressed(p,**arr);sizes[method]=p.stat().st_size
 rows=[]
 for task in HELD:
  x=xa[task];truth=f(x,amps[task],*task[:1],PHASES[task[1]])
  aa,kk,ph=codes[task];pred=f(x,aa,int(kk),ph)
  curve=curves[task];gridpred=interp(curve,x)
  for method,yhat in [('shared_mirror_decoder',pred),('native_periodic_activation',pred),('independent_sampled_curves',gridpred)]:
   ops=512*(10 if method=='shared_mirror_decoder' else 10 if method=='native_periodic_activation' else 8)+fitwork[task]
   rows.append({'seed':seed,'task':[task[0],task[1]],'method':method,'serialized_bytes':sizes[method],'heldout_nrmse':nrmse(truth,yhat),'support_points':64,'support_candidate_evaluations':4*64 if method!='independent_sampled_curves' else 0,'active_compute_proxy':ops,'optimizer_updates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();start=time.perf_counter();rows=[]
 for seed in a.seeds:rows.extend(run(seed))
 t=time.perf_counter()-start
 for x in rows:x['wall_time_s']=t
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for x in rows:w.writerow(['harmonic_sines',x['seed'],x['method'],x['serialized_bytes'],x['support_points'],0,x['active_compute_proxy'],x['wall_time_s'],'audit_nrmse',x['heldout_nrmse'],'support_candidate_evaluations',x['support_candidate_evaluations'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
