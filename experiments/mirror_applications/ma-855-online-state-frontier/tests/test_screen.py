import importlib.util
from pathlib import Path
import unittest
import numpy as np
p=Path(__file__).parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('r',p);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
class TestFrontier(unittest.TestCase):
 def test_ridge_attainability_from_support_only(self):
  rng=np.random.default_rng(5);A=rng.normal(size=(r.DO,r.DI));X=rng.normal(size=(64,r.DI));Y=X@A.T;Q=rng.normal(size=(32,r.DI));Yq=Q@A.T
  ep=(X,Y,Q,Yq);base=np.zeros_like(A);basis=np.zeros((r.R,r.DO,r.DI))
  state,e,*_=r.adapt_one(ep,base,basis,'ridge',.1)
  self.assertLess(e,1e-8)
 def test_query_labels_do_not_change_adaptation_state(self):
  base,basis,teacher,_,_,_=r.train_shared();ep=r.episodes(73,1,teacher)[0]
  a=r.adapt_one(ep,base,basis,'mirror',.29)[0]
  ep2=(ep[0],ep[1],ep[2],ep[3]+100.)
  b=r.adapt_one(ep2,base,basis,'mirror',.29)[0]
  np.testing.assert_array_equal(a,b)
 def test_same_state_object_reset_removes_episode_history(self):
  base,basis,teacher,_,_,_=r.train_shared();episodes=r.episodes(89,2,teacher);state=__import__('numpy').zeros(r.R)
  r.adapt_one(episodes[0],base,basis,'mirror',.29,state)
  state.fill(0)
  reused=r.adapt_one(episodes[1],base,basis,'mirror',.29,state)[0].copy()
  fresh=r.adapt_one(episodes[1],base,basis,'mirror',.29,__import__('numpy').zeros(r.R))[0]
  __import__('numpy').testing.assert_array_equal(reused,fresh)
 def test_native_lowrank_control_aliases_mirror(self):
  x=r.score(73,.1,.29,3.2)['methods']
  self.assertEqual(x['mirror']['nrmse'],x['native_lowrank']['nrmse'])
  self.assertEqual(x['mirror']['state_bytes'],x['native_lowrank']['state_bytes'])
if __name__=='__main__':unittest.main()
