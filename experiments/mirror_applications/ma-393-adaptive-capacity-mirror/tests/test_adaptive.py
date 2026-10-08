import sys,unittest,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run import METHODS,Model,make_data,state_arrays,pack,load
class AdaptiveTests(unittest.TestCase):
 def test_frequency_skew_and_balanced_evaluation(self):
  d=make_data(39301);self.assertEqual(len(d['train']['tokens']),32768);tokens=d['train']['tokens'];self.assertGreater(float((tokens<32).float().mean()),.65);self.assertAlmostEqual(float((d['test']['tokens']<32).float().mean()),.25,delta=.01)
 def test_all_methods_emit_finite_logits(self):
  t=torch.arange(128)
  for method in METHODS:self.assertTrue(torch.isfinite(Model(method,39301)(t)).all())
 def test_payload_is_deterministic_and_reloadable(self):
  m=Model('mirror_angle',39301);a=state_arrays(m)
  with tempfile.TemporaryDirectory() as d:
   x,h=pack(a,Path(d)/'x.npz');y,j=pack(a,Path(d)/'y.npz');self.assertEqual((x,h),(y,j))
  restored=load('mirror_angle',39301,a);t=torch.arange(128);self.assertTrue(torch.allclose(m(t),restored(t),atol=2e-4,rtol=2e-4))
if __name__=='__main__':unittest.main()
