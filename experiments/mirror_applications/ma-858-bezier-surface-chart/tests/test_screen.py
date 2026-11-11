import sys,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class SurfaceTests(unittest.TestCase):
 def test_bernstein_basis_partition_and_corners(self):
  for u,v in [(0,0),(1,0),(0,1),(1,1),(.3,.7)]:
   w=run.bern(u,v);self.assertAlmostEqual(float(w.sum()),1.,places=6)
   if (u,v)==(0,0):self.assertEqual(float(w[0,0]),1.)
   if (u,v)==(1,1):self.assertEqual(float(w[2,2]),1.)
 def test_mirror_coordinate_roundtrip(self):
  for u,v in run.COORDS:
   m=2*torch.tensor([u,v])-1;d=(m+1)/2;self.assertTrue(torch.allclose(d,torch.tensor([u,v]),atol=1e-7,rtol=0))
 def test_all_nine_control_points_receive_gradient(self):
  torch.manual_seed(17);nets=[run.Net() for _ in range(9)];x=torch.randn(16,64);y=torch.randint(0,10,(16,));w=run.bern(.37,.62);states=[dict(n.named_parameters()) for n in nets];mix={k:sum(w[i,j]*states[3*i+j][k] for i in range(3) for j in range(3)) for k in states[0]};loss=nn.functional.cross_entropy(torch.func.functional_call(nets[0],mix,(x,)),y);loss.backward()
  for n in nets:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in n.parameters()))
if __name__=='__main__':unittest.main()
