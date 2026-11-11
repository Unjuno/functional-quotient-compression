import sys,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class PrivateResidualTests(unittest.TestCase):
 def test_flatten_unflatten(self):
  m=run.Net();v=run.flat(m);p=run.unflat(m,v)
  for n,x in m.named_parameters():self.assertTrue(torch.equal(x,p[n]))
 def test_private_rank_one_shapes_and_forward(self):
  m=run.Net();p=m.state_dict();A=torch.randn(10,1);B=torch.randn(1,run.WIDTH);self.assertEqual((A@B).shape,p['fc2.weight'].shape);q=dict(p);q['fc2.weight']=q['fc2.weight']+A@B;self.assertEqual(tuple(q['fc2.weight'].shape),(10,run.WIDTH))
 def test_actual_serialized_rank3_and_rank6_basis_bytes_differ(self):
  torch.manual_seed(1);m=run.Net();b=run.flat(m);basis=torch.randn(6,len(b));c3=torch.zeros(3);c6=torch.zeros(6);n3,_=run.payload(m,b,basis[:3],c3);n6,_=run.payload(m,b,basis[:6],c6);self.assertGreater(n6-n3,3*len(b)*4-100)
if __name__=='__main__':unittest.main()
