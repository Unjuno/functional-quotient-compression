import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import ContextNet,Independent,rotate
class Tests(unittest.TestCase):
 def test_rotation_zero(self):
  h=torch.randn(2,16);self.assertTrue(torch.equal(h,rotate(h,torch.zeros(2,4))))
 def test_dynamic_models(self):
  x=torch.randn(7,16);c=torch.tensor([0,1,2,3,0,1,2])
  for method in ('shared','mirror_gen','film_gen','static'):self.assertEqual(tuple(ContextNet(method)(x,c).shape),(7,4))
 def test_independent_dispatch(self):self.assertEqual(tuple(Independent()(torch.randn(7,16),torch.tensor([0,1,2,3,0,1,2])).shape),(7,4))
if __name__=='__main__':unittest.main()
