import csv,importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma344",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_heterogeneous_rank_prefix_structure():
 W,B,A1,A2,r,p,mat,x,truth,Bo,Ao,xo,to=m.world(34411)
 assert set(r.tolist())=={1,2,4}
 for i,ri in enumerate(r):
  aa=np.cos(p[i,:ri,None])*A1[:ri]+np.sin(p[i,:ri,None])*A2[:ri]
  pred=x[i]@W.T+(x[i]@aa.T)@B[:,:ri].T
  assert m.nerr(pred,truth[i])<1e-10

def test_private_rank_four_state_restores_offorbit():
 W,B,A1,A2,r,p,mat,x,truth,Bo,Ao,xo,to=m.world(34412)
 assert m.nerr(xo@W.T+(xo@Ao.T)@Bo.T,to)<1e-10
 assert m.nerr(xo@W.T,to)>0.1

def test_native_phase_payload_equals_mirror_in_development():
 rows=list(csv.DictReader((Path(__file__).parents[1]/"artifacts"/"development.csv").open()))
 for seed in ('34401','34402'):
  a=next(r for r in rows if r['seed']==seed and r['method']=='mirror_segment_phase_private')
  b=next(r for r in rows if r['seed']==seed and r['method']=='native_scalar_phase_private')
  assert a['sha256']==b['sha256'] and a['payload_bytes']==b['payload_bytes']
