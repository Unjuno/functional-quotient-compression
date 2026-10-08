import sys,unittest,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from run import METHODS,Model,held_split,make_data,arrays_for,pack,load_model
class DomainTests(unittest.TestCase):
 def test_exact_holdout_and_marginals_seen(self):
  held=held_split(39201).reshape(64,4);self.assertEqual(int(held.sum()),64);self.assertTrue((~held).any(0).all());self.assertTrue((~held).any(1).all())
 def test_data_reproducible_and_train_excludes_holdout(self):
  a,b=make_data(39201),make_data(39201);self.assertTrue(torch.equal(a['train']['token'],b['train']['token']));self.assertFalse(a['train']['held'].any());self.assertTrue(a['test']['held'].any())
 def test_fp16_payload_deterministic_and_reloads(self):
  model=Model('mirror_domain',39201);arrays=arrays_for(model)
  with tempfile.TemporaryDirectory() as d:
   x,h=pack(arrays,Path(d)/'x.npz');y,j=pack(arrays,Path(d)/'y.npz');self.assertEqual((x,h),(y,j))
  restored=load_model('mirror_domain',39201,arrays);t=torch.arange(64);dom=torch.arange(64)%4;self.assertTrue(torch.allclose(model(t,dom),restored(t,dom),atol=2e-4,rtol=2e-4))
if __name__=='__main__':unittest.main()
