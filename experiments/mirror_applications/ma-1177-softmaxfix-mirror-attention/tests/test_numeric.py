import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from core import fixed_ce, standard_ce, standard_soft_ce, fixed_soft_ce, log_softmax_fixed, house, TinyAttention, fp64_cross_entropy

class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fp32_high_margin_target_gradient(self):
        z = torch.tensor([[9., -9., -9.]], requires_grad=True)
        c = torch.tensor([0])
        g_old = torch.autograd.grad(standard_ce(z, c),z,retain_graph=True)[0]
        g_fix = torch.autograd.grad(fixed_ce(z,c),z)[0]
        self.assertEqual(float(g_old[0,0]),0.0)
        self.assertLess(float(g_fix[0,0]),-3.0e-8)
        self.assertLess(abs(float(g_fix.sum())),1e-13)
        self.assertLess(abs(float(fixed_ce(z,c).item())-3.0459958e-8),1e-13)

    def test_bf16_high_margin(self):
        z=torch.tensor([[7.,0.,0.]],dtype=torch.bfloat16,requires_grad=True)
        c=torch.tensor([0]);g0=torch.autograd.grad(standard_ce(z,c),z,retain_graph=True)[0];g1=torch.autograd.grad(fixed_ce(z,c),z)[0]
        self.assertEqual(float(g0[0,0]),0.)
        self.assertLess(float(g1[0,0]),-0.0015)

    def test_equal_logits_teacher_student(self):
        for dtype in (torch.float32,torch.bfloat16,torch.float64):
            x=torch.tensor([[2.,1.,0.],[9.,-9.,-9.]],dtype=dtype,requires_grad=True)
            target=x.detach().clone()
            gfix=torch.autograd.grad(fixed_soft_ce(x,target),x)[0]
            self.assertTrue((gfix==0).all().item(),(dtype,gfix))

    def test_near_teacher_matches_double(self):
        s=torch.tensor([[2. - 1e-7, 1. + 1e-7, 0.]],dtype=torch.float32,requires_grad=True)
        t=torch.tensor([[2.,1.,0.]],dtype=torch.float32)
        g0=torch.autograd.grad(standard_soft_ce(s,t),s,retain_graph=True)[0]
        g1=torch.autograd.grad(fixed_soft_ce(s,t),s)[0]
        sref=s.detach().double().requires_grad_(); tref=t.double()
        gp=torch.autograd.grad(standard_soft_ce(sref,tref),sref)[0].float()
        self.assertLess(torch.linalg.vector_norm(g1-gp),torch.linalg.vector_norm(g0-gp))

    def test_attention_query_and_key_same_householder_cancels(self):
        torch.manual_seed(123)
        q=torch.randn(5,8,dtype=torch.float64);k=torch.randn(11,8,dtype=torch.float64);u=torch.randn(8,dtype=torch.float64)
        self.assertLess(float(((house(q,u)@house(k,u).T)-(q@k.T)).abs().max()),1e-13)

    def test_identical_initializations(self):
        for name in ('mirror','film','independent'):
            a=TinyAttention(name,1801,12.0);b=TinyAttention(name,1801,12.0)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))

    def test_ordinary_hard_ce_normal_regime(self):
        torch.manual_seed(123)
        for _ in range(10):
            z=(torch.randn(3,12)*2).requires_grad_()
            t=torch.randint(0,12,(3,))
            self.assertLess(abs(standard_ce(z,t).item()-fixed_ce(z,t).item()),5e-7)

    def test_logsoftmax_fixed_properties(self):
        z=torch.tensor([[9.,-9.,-9.]],requires_grad=True)
        logp=log_softmax_fixed(z)
        g=torch.autograd.grad(logp[0,0],z)[0]
        self.assertGreater(float(g[0,0]),2.9e-8)
        self.assertLess(abs(float(g.sum())),1e-12)

if __name__=='__main__':unittest.main()
