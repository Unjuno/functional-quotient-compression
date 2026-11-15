import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma1040_run',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_givens_aliases_matched_linear_control():
 x=np.random.default_rng(1040).normal(size=(16,768)).astype(np.float32)
 for theta in m.GRID:
  np.testing.assert_array_equal(m.givens(x,theta),m.linear(x,theta))

def test_rotation_preserves_query_norm():
 x=np.random.default_rng(1041).normal(size=(9,768)).astype(np.float32)
 for theta in m.GRID:
  np.testing.assert_allclose(np.linalg.norm(m.givens(x,theta),axis=1),np.linalg.norm(x,axis=1),rtol=2e-7,atol=2e-7)

def test_hash_split_disjoint_and_total():
 q=[f'q{i}' for i in range(1000)]
 d=[]; a=[]
 import hashlib
 for qid in q:
  (d if hashlib.sha256((m.SALT+qid).encode()).digest()[0]<128 else a).append(qid)
 assert set(d).isdisjoint(a) and len(d)+len(a)==len(q)
 assert 400<len(d)<600
