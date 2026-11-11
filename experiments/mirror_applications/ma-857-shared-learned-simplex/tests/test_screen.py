import sys, unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

class SimplexTests(unittest.TestCase):
 def test_native_and_mirror_barycentric_coordinates_match(self):
  torch.manual_seed(2); nets=[run.Net() for _ in range(3)]; states=[n.state_dict() for n in nets]; x=torch.randn(16,64)
  for a in run.ALPHAS:
   m=2*a-1; decoded=(m+1)/2
   p={k:sum(float(a[i])*states[i][k] for i in range(3)) for k in states[0]}
   q={k:sum(float(decoded[i])*states[i][k] for i in range(3)) for k in states[0]}
   self.assertTrue(torch.allclose(torch.func.functional_call(nets[0],p,(x,)),torch.func.functional_call(nets[0],q,(x,)),atol=5e-6,rtol=0))
 def test_barycentric_training_reaches_all_three_vertices(self):
  torch.manual_seed(8);nets=[run.Net() for _ in range(3)]; x=torch.randn(10,64);y=torch.randint(0,10,(10,));a=torch.tensor([.2,.3,.5]);states=[dict(n.named_parameters()) for n in nets];mix={k:sum(a[i]*states[i][k] for i in range(3)) for k in states[0]};loss=nn.functional.cross_entropy(torch.func.functional_call(nets[0],mix,(x,)),y);loss.backward()
  for n in nets:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in n.parameters()))
 def test_data_split_has_equal_domain_counts_and_stratification(self):
  (x,y,d),_,_=run.get_data();self.assertEqual(len(x),3*len(y)//3);self.assertEqual(torch.bincount(d).tolist(),[len(d)//3]*3)
if __name__=='__main__':unittest.main()
