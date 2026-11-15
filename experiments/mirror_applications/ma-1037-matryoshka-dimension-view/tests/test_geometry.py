import importlib.util
from pathlib import Path
import unittest
import numpy as np
p=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
spec=importlib.util.spec_from_file_location('ma1037',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class TestMatryoshkaView(unittest.TestCase):
 def test_boundary_givens_equals_normalized_linear_control(self):
  x=np.random.default_rng(7).normal(size=(13,m.D)).astype(np.float32)
  for d,theta in [(64,.37),(128,-.9),(512,.1),(768,0.)]:
   a=m.givens_prefix(x,d,theta); b=m.linear_boundary_prefix(x,d,[np.cos(theta),-np.sin(theta)])
   np.testing.assert_array_equal(a,b)
 def test_full_dimension_orthogonal_gauge_keeps_cosine(self):
  x=np.random.default_rng(11).normal(size=(9,m.D)).astype(np.float32)
  y=np.random.default_rng(12).normal(size=(15,m.D)).astype(np.float32)
  x=m.normalize(x); y=m.normalize(y)
  # Exact all-dimension orthogonal rotation at the final coordinate plane.
  c=np.cos(.73); s=np.sin(.73)
  xa=x.copy(); ya=y.copy()
  a=xa[:,-1].copy();b=xa[:,0].copy(); xa[:,-1]=c*a-s*b; xa[:,0]=s*a+c*b
  a=ya[:,-1].copy();b=ya[:,0].copy(); ya[:,-1]=c*a-s*b; ya[:,0]=s*a+c*b
  np.testing.assert_allclose(x@y.T,xa@ya.T,rtol=2e-6,atol=2e-6)
 def test_metric_known_answer(self):
  q=np.eye(2,dtype=np.float32); d=np.eye(2,dtype=np.float32)
  met=m.ranking_metrics(q,d,['q0','q1'],['d0','d1'],{'q0':{'d0':1},'q1':{'d1':1}})
  self.assertEqual(met['nDCG@10'],1.); self.assertEqual(met['Recall@10'],1.); self.assertEqual(met['MRR@10'],1.)
 def test_serialized_npy_round_trip_and_actual_bytes(self):
  import tempfile
  with tempfile.TemporaryDirectory() as td:
   x=np.random.default_rng(4).normal(size=(7,64)).astype(np.float32)
   n,digest=m.save_npy(Path(td)/'vectors.npy',x)
   self.assertEqual(n,(Path(td)/'vectors.npy').stat().st_size)
   np.testing.assert_array_equal(np.load(Path(td)/'vectors.npy',allow_pickle=False),x)
   self.assertEqual(len(digest),64)
if __name__=='__main__': unittest.main()
