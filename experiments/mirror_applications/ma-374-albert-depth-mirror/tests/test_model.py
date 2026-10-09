import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import DepthNet,METHODS,rotate
class Tests(unittest.TestCase):
 def test_givens_zero(self):
  x=torch.randn(4,64);self.assertTrue(torch.equal(x,rotate(x,torch.zeros(4))))
 def test_all_methods_forward(self):
  x=torch.randn(3,64)
  for m in METHODS:self.assertEqual(tuple(DepthNet(m)(x).shape),(3,10))
 def test_sharing_parameter_counts(self):
  self.assertLess(sum(p.numel() for p in DepthNet('tied').parameters()),sum(p.numel() for p in DepthNet('untied').parameters()))
if __name__=='__main__':unittest.main()
