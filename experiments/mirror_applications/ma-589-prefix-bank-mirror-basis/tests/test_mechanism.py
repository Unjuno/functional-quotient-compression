import importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).parents[1]/'source'/'run_experiment.py';spec=importlib.util.spec_from_file_location('ma589',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_kv_layout_has_one_row_per_prefix_token():
 k=np.zeros((1,m.H,9,m.D),np.float32);v=np.ones_like(k);cache=[[k,v] for _ in range(m.L)]
 assert m.tomat(cache,0).shape==(9,m.F)
def test_rank_four_shared_view_reconstructs_native_representation():
 rng=np.random.default_rng(589);x=rng.normal(size=(m.PFX,m.F)).astype(np.float32);mu,e=m.pca(x,8);res=x-(((x-mu)@e)@e.T+mu)
 bases=np.stack([m.pca(res.reshape(-1,m.ROLE,m.D)[:,r],2)[1] for r in range(m.ROLE)])
 def decode():
  z=((x-mu)@e).astype(np.float16).astype(np.float32);base=z@e.T+mu;rr=(x-base).reshape(-1,m.ROLE,m.D)
  c=np.stack([rr[:,r]@bases[r] for r in range(m.ROLE)],1).astype(np.float16).astype(np.float32)
  return base+np.stack([c[:,r]@bases[r].T for r in range(m.ROLE)],1).reshape(x.shape)
 assert np.array_equal(decode(),decode())
