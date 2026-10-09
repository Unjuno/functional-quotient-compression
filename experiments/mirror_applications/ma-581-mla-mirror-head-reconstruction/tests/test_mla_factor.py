import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'source'/'run_experiment.py';s=importlib.util.spec_from_file_location('ma581',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_kv_matrix_roundtrip():
 cache=[[np.random.default_rng(1).normal(size=(1,m.H,5,m.D)).astype(np.float32),np.random.default_rng(2).normal(size=(1,m.H,5,m.D)).astype(np.float32)] for _ in range(m.L)]
 for layer in range(m.L):
  x=m.to_matrix(cache,layer);k,v=m.from_matrix(x);np.testing.assert_allclose(k.numpy()[0],cache[layer][0][0]);np.testing.assert_allclose(v.numpy()[0],cache[layer][1][0])
def test_pca_reconstructs_low_rank():
 r=np.random.default_rng(3);a=r.normal(size=(80,10)).astype(np.float32);a[:,5:]=a[:,:5]@r.normal(size=(5,5)).astype(np.float32)
 mu,e=m.pca(a,5);z=(a-mu)@e;rec=z@e.T+mu;assert rec.shape==a.shape;assert np.mean((rec-a)**2)<1e-5
def test_shared_residual_coefficients_have_head_role_axis():
 assert m.F==m.H*m.KV*m.D==1024
