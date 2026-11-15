import importlib.util, math, unittest
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'source'/'run_experiment.py'
s=importlib.util.spec_from_file_location('ma1036',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
class TestGeometry(unittest.TestCase):
 def test_givens_preserves_norm_and_distances(self):
  x=np.random.default_rng(1).normal(size=(17,m.D)).astype('float32'); y=np.random.default_rng(2).normal(size=(19,m.D)).astype('float32')
  a=m.givens(x,.37); b=m.givens(y,.37)
  np.testing.assert_allclose(np.linalg.norm(a,axis=1),np.linalg.norm(x,axis=1),rtol=1e-6,atol=1e-5)
  np.testing.assert_allclose(np.sum((a[:,None]-b[None,:])**2,axis=2),np.sum((x[:,None]-y[None,:])**2,axis=2),rtol=2e-6,atol=2e-5)
 def test_rotation_basis_is_orthogonal(self):
  R=m.rotation(1036)
  np.testing.assert_allclose(R@R.T,np.eye(m.D),atol=2e-5)
 def test_faiss_index_serialization_roundtrip(self):
  train,db,qs=m.world(17,ntrain=64,ndb=80,nq=5)
  idx=m.index_for(train,db); d0,i0=idx.search(qs[0],m.K)
  restored=m.faiss.deserialize_index(m.faiss.serialize_index(idx)); d1,i1=restored.search(qs[0],m.K)
  np.testing.assert_array_equal(i0,i1); np.testing.assert_allclose(d0,d1,rtol=0,atol=0)
 def test_repeat_world_is_deterministic(self):
  a=m.world(123,ntrain=8,ndb=9,nq=4); b=m.world(123,ntrain=8,ndb=9,nq=4)
  for x,y in zip(a[:2],b[:2]): np.testing.assert_array_equal(x,y)
  for x,y in zip(a[2],b[2]): np.testing.assert_array_equal(x,y)
if __name__=='__main__': unittest.main()
