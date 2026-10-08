import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import WIDTHS,NestedNet,givens
class Tests(unittest.TestCase):
 def test_zero_angles_identity_and_norm(self):
  x=torch.randn(8,16);self.assertTrue(torch.equal(givens(x,torch.zeros(4)),x));y=givens(x,torch.tensor([.1,-.2,.3,-.4]));self.assertLess(float((x[:,:8].square().sum()-y[:,:8].square().sum()).abs()),2e-5)
 def test_nested_widths(self):
  m=NestedNet();x=torch.randn(5,64)
  for w in WIDTHS:self.assertEqual(tuple(m(x,w).shape),(5,10))
 def test_layer_codes_are_per_width(self):
  self.assertEqual(len(WIDTHS),3)
if __name__=='__main__':unittest.main()
