import csv,importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/"source"/"run.py";spec=importlib.util.spec_from_file_location("ma349",P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_rank1_factor_and_mirror_codes_generate_same_two_posterior_modes():
 rng,w,j,d,mode,prob,x,y,ptrue,wm,wp=m.world(34911);scale=d/w[j]
 for eps,expected in zip(mode,(wm,wp)):
  w_bnn=w.copy();w_bnn[j]+=eps*w[j]*scale
  w_mirror=w.copy();w_mirror[j]+=eps*d
  assert np.max(np.abs(w_bnn-expected))<1e-6 and np.max(np.abs(w_mirror-expected))<1e-6

def test_two_mode_predictive_distribution_is_calibrated_teacher():
 rng,w,j,d,mode,prob,x,y,ptrue,wm,wp=m.world(34912)
 p=.5*m.sigmoid(x@wm)+.5*m.sigmoid(x@wp)
 assert np.max(np.abs(p-ptrue))<1e-7

def test_development_controls_and_payload_replay_rows_exist():
 p=Path(__file__).parents[1]/"artifacts"/"development.csv";assert p.exists();rows=list(csv.DictReader(p.open()));assert len(rows)==10
 for seed in ('34901','34902'):
  a=next(r for r in rows if r['seed']==seed and r['method']=='mirror_scalar_posterior');b=next(r for r in rows if r['seed']==seed and r['method']=='direct_sparse_scalar_posterior')
  assert a['sha256']==b['sha256']
