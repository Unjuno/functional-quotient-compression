import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import unittest
import torch
from model import *

class TestModels(unittest.TestCase):
  def test_counts(self):
    for w in WIDTHS:
      n=12*w+4
      assert param_count(new_student('mirror',w,1))==n
      assert param_count(new_student('film',w,1))==n
      independent=param_count(new_student('independent',w,1))
      assert abs(independent/n-1)<0.08
  def test_householder(self):
    v=torch.randn(3,16,dtype=torch.double)
    h=householder(v)
    assert torch.allclose(h.transpose(-1,-2)@h,torch.eye(16,dtype=torch.double),atol=1e-12)
    assert torch.allclose(h,h.transpose(-1,-2))
  def test_fast_reflection_vs_matrix(self):
    v=torch.randn(5,16,dtype=torch.double)
    z=torch.randn(5,16,dtype=torch.double)
    assert torch.allclose(reflect(z,v),(householder(v)@z.unsqueeze(-1)).squeeze(-1),atol=1e-12)
  def test_teacher_as_dense(self):
    base=DenseMLP(16)
    directions=torch.randn(3,16)
    teachers=teacher_mirror_family(base,directions)
    with torch.no_grad():
      m=SharedStudent(16,'mirror')
      m.up.load_state_dict(base.up.state_dict());m.down.load_state_dict(base.down.state_dict())
      m.codes.copy_(directions)
      x=torch.randn(64,4);rule=torch.randint(3,(64,))
      assert (m(x,rule)-teacher_outputs(teachers,x,rule)).abs().max()<2e-6
  def test_film_teacher_exact(self):
    base=DenseMLP(16);code=torch.randn(3,16)
    ts=teacher_film_family(base,code)
    m=SharedStudent(16,'film')
    m.up.load_state_dict(base.up.state_dict());m.down.load_state_dict(base.down.state_dict())
    with torch.no_grad():m.codes.copy_(code)
    x=torch.randn(64,4);r=torch.randint(3,(64,))
    assert (m(x,r)-teacher_outputs(ts,x,r)).abs().max()<2e-6
  def test_grad(self):
    for kind in ('mirror','film','independent','shared'):
      m=new_student(kind,16,22);x=torch.randn(30,4);r=torch.randint(3,(30,))
      m(x,r).square().mean().backward()
      assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  def test_teacher_outputs_stable(self):
    base=DenseMLP(16);ts=teacher_mirror_family(base,torch.randn(3,16))
    x=torch.randn(24,4);r=torch.randint(3,(24,))
    y=teacher_outputs(ts,x,r)
    assert y.shape==(24,4)

if __name__=='__main__':unittest.main()
