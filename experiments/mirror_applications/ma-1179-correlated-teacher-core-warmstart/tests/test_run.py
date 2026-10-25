"""MA-1179 fast mechanism checks without touching sealed Fresh samples."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import unittest
import torch
from torch.nn import functional as F
from run import *

class TestMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_physical_parameter_count(self):
        for typ in ('mirror','film','rank1'):self.assertEqual(count(SharedStudent(typ)),196)
        self.assertEqual(count(SharedStudent('shared')),148)
        self.assertEqual(count(IndependentStudent()),201)
    def test_householder_involution(self):
        x=torch.randn(5,H);v=torch.randn(5,H)
        self.assertLess((reflection(reflection(x,v),v)-x).abs().max().item(),1e-5)
    def test_same_core_initialization(self):
        a=new_student('mirror',123);b=new_student('film',123);c=new_student('rank1',123)
        self.assertTrue(torch.equal(a.up.weight,b.up.weight))
        self.assertTrue(torch.equal(b.down.weight,c.down.weight))
    def test_oracle_exact_positive(self):
        torch.manual_seed(33)
        base=DenseMLP();group,dirs=reflect_teachers(base,101)
        m=SharedStudent('mirror');m.up.load_state_dict(base.up.state_dict());m.down.load_state_dict(base.down.state_dict())
        with torch.no_grad():m.codes.copy_(dirs)
        x,r=draw(6123,768);o=output(group,x,r)
        self.assertLess((m(x,r).detach()-o).square().mean().item(),1e-12)
    def test_teacher_independent_init(self):
        a=DenseMLP();b=DenseMLP()
        self.assertFalse(torch.equal(a.up.weight,b.up.weight))
    def test_source_task_variation(self):
        x,_=draw(10,128)
        a=teacher_normal(x,0,101);b=teacher_normal(x,1,101)
        self.assertGreater(F.mse_loss(a,b).item(),.01)
    def test_finite_gradients(self):
        for kind in ('mirror','film','rank1','shared','independent'):
            m=new_student(kind,23);x,r=draw(14,128);loss=m(x,r).square().mean();loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))

if __name__=='__main__':unittest.main()
