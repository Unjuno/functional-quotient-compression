import csv,importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma342",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_unseen_client_factor_family_is_rank_two():
 W,b,a1,a2,aoff,z,zoff,t,off,sup,xev,truth,offdata=m.world(34211)
 for i in (0,17,48,63):
  expected=W+b@(z[i,0]*a1+z[i,1]*a2)
  assert np.max(np.abs(expected-t[i]))<1e-6

def test_private_right_factor_recovers_offorbit_function():
 W,b,a1,a2,aoff,z,zoff,t,off,sup,xev,truth,offdata=m.world(34212)
 right=np.linalg.lstsq(b,off-W,rcond=None)[0]
 assert np.max(np.abs(W+b@right-off))<1e-5

def test_native_phase_and_mirror_payload_hashes_match():
 p=Path(__file__).parents[1]/"artifacts"/"development.csv";rows=list(csv.DictReader(p.open()))
 for seed in ('34201','34202'):
  a=next(r for r in rows if r['seed']==seed and r['method']=='mirror_phase_with_private')
  b=next(r for r in rows if r['seed']==seed and r['method']=='native_phase_with_private')
  assert a['sha256']==b['sha256'] and a['payload_bytes']==b['payload_bytes']
