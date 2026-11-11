import sys,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class ProjectionTests(unittest.TestCase):
 def test_flatten_unflatten_roundtrip(self):
  m=run.Net();v=run.flat(m);p=run.unflat(m,v)
  for n,t in m.named_parameters():self.assertTrue(torch.equal(t,p[n]))
 def test_native_and_mirror_code_views_are_identical(self):
  torch.manual_seed(8);m=run.Net();b=run.flat(m);B=torch.randn(3,len(b));c=torch.randn(3);x=torch.randn(12,64)
  native=run.unflat(m,b+c@B);mirror_code=c.clone();mirror=run.unflat(m,b+mirror_code@B)
  a=torch.func.functional_call(m,native,(x,));z=torch.func.functional_call(m,mirror,(x,));self.assertTrue(torch.equal(a,z))
 def test_fewshot_code_receives_gradient(self):
  torch.manual_seed(4);m=run.Net();b=run.flat(m);B=torch.randn(3,len(b));c=torch.zeros(3,requires_grad=True);x=torch.randn(10,64);y=torch.randint(0,10,(10,));loss=nn.functional.cross_entropy(torch.func.functional_call(m,run.unflat(m,b+c@B),(x,)),y);loss.backward();self.assertGreater(float(c.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
