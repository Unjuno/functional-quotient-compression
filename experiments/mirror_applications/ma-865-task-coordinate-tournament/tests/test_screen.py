import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class TournamentTests(unittest.TestCase):
 def test_flatten_unflatten_preserves_parameters(self):
  m=run.Net();v=run.flat(m);p=run.unflat(m,v)
  for n,x in m.named_parameters():self.assertTrue(torch.equal(x,p[n]))
 def test_rank4_svd_reconstructs_four_delta_vectors(self):
  torch.manual_seed(3);D=torch.randn(4,run.Net().fc1.weight.numel()+run.Net().fc1.bias.numel()+run.Net().fc2.weight.numel()+run.Net().fc2.bias.numel());_,_,vh=torch.linalg.svd(D,full_matrices=False);B=vh[:4];R=D@B.T;self.assertTrue(torch.allclose(R@B,D,atol=2e-5,rtol=1e-5))
 def test_native_and_mirror_codes_are_functionally_identical(self):
  torch.manual_seed(6);m=run.Net();b=run.flat(m);B=torch.randn(2,len(b));c=torch.randn(2);x=torch.randn(11,64);a=torch.func.functional_call(m,run.unflat(m,b+c@B),(x,));mirrorc=c.clone();z=torch.func.functional_call(m,run.unflat(m,b+mirrorc@B),(x,));self.assertTrue(torch.equal(a,z))
if __name__=='__main__':unittest.main()
