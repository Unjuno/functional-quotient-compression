import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import CONFIGS,MIXED,Nested3,transform
class Tests(unittest.TestCase):
 def test_config_partition(self):self.assertEqual((len(CONFIGS),len(MIXED)),(27,24))
 def test_identity_rotation(self):
  x=torch.randn(5,16);self.assertTrue(torch.equal(x,transform(x,'factor_mirror',torch.zeros(4))))
 def test_mixed_forward_shapes(self):
  m=Nested3();x=torch.randn(3,64)
  for c in MIXED:self.assertEqual(tuple(m(x,c).shape),(3,10))
if __name__=='__main__':unittest.main()
