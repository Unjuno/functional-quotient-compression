import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import Net,transform,METHODS
class Tests(unittest.TestCase):
 def test_givens_zero_identity(self):
  h=torch.randn(3,16);self.assertTrue(torch.equal(h,transform(h,'mirror',{'angles':torch.zeros(4)})))
 def test_conditioned_logits(self):
  m=Net();x=torch.randn(5,64)
  for method,code in [('shared',None),('mirror',{'angles':torch.zeros(4)}),('film4',{'scale':torch.ones(4)}),('film8',{'scale':torch.ones(4),'bias':torch.zeros(4)})]:self.assertEqual(tuple(m(x,method,code).shape),(5,10))
 def test_method_set(self):self.assertEqual(len(METHODS),6)
if __name__=='__main__':unittest.main()
