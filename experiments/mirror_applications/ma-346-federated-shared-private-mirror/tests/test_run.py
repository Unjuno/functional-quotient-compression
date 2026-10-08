import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma346",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_private_residual_rank_and_heterogeneity_sweep():
 rng,W,U,V,ph,common,res,p,q,x=m.world(34611)
 assert res.shape==(m.NC,m.D,m.D)
 for i in range(m.NC):assert np.linalg.matrix_rank(res[i],tol=1e-5)<=1

def test_shared_plus_private_reconstructs_all_client_functions():
 rng,W,U,V,ph,common,res,p,q,x=m.world(34612);targets=common.copy();targets[:16]+=res[:16]
 for i in range(m.NC):
  pred=x[i]@targets[i].T
  assert np.max(np.abs(pred-x[i]@targets[i].T))==0

def test_native_scalar_phase_state_has_same_contract():
 p=Path(__file__).parents[1]/"artifacts"/"development.csv"
 if not p.exists():return
 import csv
 rows=list(csv.DictReader(p.open()))
 for seed in ('34601','34602'):
  for rate in (0,.25,.5,.75,1.):
   a=next(r for r in rows if r['seed']==seed and float(r['heterogeneity_rate'])==rate and r['method']=='mirror_rank1_private')
   b=next(r for r in rows if r['seed']==seed and float(r['heterogeneity_rate'])==rate and r['method']=='native_scalar_phase_private')
   assert a['payload_bytes']==b['payload_bytes'] and a['sha256']==b['sha256']
