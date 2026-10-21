import sys,os,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import VirtualGraph,DenseControl,new_model,EDGES,fixed_program
from data import dataset,teacher

class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_shape_and_determinism(self):
        x=torch.randn(5,8);m=new_model('mirror',51)
        self.assertEqual(tuple(m(x).shape),(5,8))
        self.assertTrue(torch.equal(m(x),m(x)))
        self.assertEqual(EDGES,32)
    def test_branch_and_gates(self):
        x=torch.randn(7,8);m=new_model('mirror',52);before=m(x)
        m.cut([0,1,2,3])
        self.assertEqual(m.active_count(),28)
        self.assertGreater((before-m(x)).abs().max().item(),1e-9)
        self.assertEqual(m.weights().shape[0],28)
    def test_gradients(self):
        m=new_model('mirror',53);m(torch.randn(7,8)).square().mean().backward()
        for p in (m.core,m.address,m.gates):
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.norm().item(),1e-11)
    def test_controls_and_parameter_counts(self):
        m=new_model('mirror',54);a=new_model('affine',54);t=new_model('direct_tanh',54)
        self.assertEqual(sum(p.numel() for p in m.parameters()),320)
        self.assertEqual(sum(p.numel() for p in a.parameters()),320)
        self.assertLess(abs(sum(p.numel() for p in DenseControl().parameters())-320),10)
        x=torch.randn(13,8)
        self.assertGreater((m(x)-t(x)).abs().mean().item(),1e-10)
        self.assertGreater((m(x)-a(x)).abs().mean().item(),1e-10)
    def test_packet_reconstruction(self):
        m=new_model('mirror',55);m.cut([1,4,17]);p=m.packet();n=new_model('mirror',99);n.reload_packet(p)
        x=torch.randn(9,8)
        self.assertLess((n(x)-m(x)).abs().max().item(),1e-7)
        self.assertEqual(m.active_count(),29)
        self.assertLess(m.packet_bytes(),new_model('mirror',57).packet_bytes())
    def test_teacher_oracle_and_data_isolation(self):
        t=teacher('aligned_sparse_mirror',61);n=new_model('mirror',63)
        n.load_state_dict(t.state_dict())
        x=torch.randn(10,8)
        self.assertLess((t(x)-n(x)).abs().max().item(),1e-8)
        tr=dataset('aligned_sparse_mirror','train',61,100)[0]
        fr=dataset('aligned_sparse_mirror','fresh',61,100)[0]
        self.assertFalse(torch.equal(tr,fr))
    def test_no_active_edge(self):
        m=new_model('mirror',67);m.active.zero_();out=m(torch.randn(3,8))
        self.assertTrue(torch.allclose(out,torch.zeros_like(out),atol=1e-7))

if __name__=='__main__':unittest.main()
