#!/usr/bin/env python3
"""Deterministic post-run wall-clock profiling of the frozen router and rank-one fits."""
import argparse,json,time
from pathlib import Path
import numpy as np
from task_data_and_pinning import load_model
from run_experiment import fit_router,fit_rank1_controls

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--experiment-dir',required=True);a=ap.parse_args()
 model,tok,torch=load_model(a.model_dir);base=Path(a.experiment_dir)
 for seed in (54501,54502,54511,54512,54513):
  name=f'dev_{seed}' if seed<54510 else f'fresh_{seed}';root=base/'results'/name
  with np.load(root/'extract.npz',allow_pickle=False) as z:x=z['router_x'];y=z['router_y'];fvs=z['function_vectors']
  torch.manual_seed(seed);np.random.seed(seed);t=time.perf_counter();router=fit_router(x,y,torch);router_s=time.perf_counter()-t
  with np.load(root/'routed_fv.npz',allow_pickle=False) as z:
   router_max_abs_diff=max(float(np.max(np.abs(router[akey]-z[k]))) for akey,k in [('mean','router_mean'),('std','router_std'),('weight','router_weight'),('bias','router_bias')])
  manifest=json.loads((root/'split_manifest.json').read_text());t=time.perf_counter();As,Bs,_=fit_rank1_controls(model,tok,torch,fvs,manifest);rank1_s=time.perf_counter()-t
  with np.load(root/'rank1.npz',allow_pickle=False) as z:rank1_max_abs_diff=max(float(np.max(np.abs(As-z['lora_A']))),float(np.max(np.abs(Bs-z['lora_B']))))
  p=root/'metrics.json';d=json.loads(p.read_text());d['compute']['router_fit_seconds_profiled']=router_s;d['compute']['rank1_fit_seconds_profiled']=rank1_s;d['compute']['profiled_router_max_abs_parameter_diff']=router_max_abs_diff;d['compute']['profiled_rank1_max_abs_parameter_diff']=rank1_max_abs_diff
  d['compute']['profile_note']='Deterministic refit on stored features/support, exact parameters verified; elapsed time is a post-run CPU profile, not part of the original end-to-end timed run.'
  p.write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
  print(seed,json.dumps({'router_fit_seconds':router_s,'rank1_fit_seconds':rank1_s,'router_max_abs_parameter_diff':router_max_abs_diff,'rank1_max_abs_parameter_diff':rank1_max_abs_diff}))
if __name__=='__main__':main()
