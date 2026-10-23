import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import unittest
import torch
from models import FeatureModel, LoopModel, LoopOnePassControl, householder, packet_bytes
from data import generate, loop_teacher

class CoreTests(unittest.TestCase):
    def test_reflection_is_involution(self):
        torch.manual_seed(1)
        x=torch.randn(9,6,dtype=torch.float64);u=torch.randn(9,6,dtype=torch.float64)
        self.assertLess((householder(householder(x,u),u)-x).abs().max().item(),1e-12)

    def test_synthetic_model_counts(self):
        for width in (4,8,12):
            self.assertEqual(FeatureModel('shared',width).num_parameters(),8*width)
            for kind in ('film','mirror'):
                self.assertEqual(FeatureModel(kind,width).num_parameters(),11*width)
            self.assertEqual(FeatureModel('independent',width).num_parameters(),24*width)
        for steps in (1,2,3): self.assertEqual(LoopModel(steps).num_parameters(),32)
        self.assertEqual(LoopOnePassControl().num_parameters(),32)

    def test_serialized_size_includes_code(self):
        for kind in ('mirror','film','independent','shared'):
            m=FeatureModel(kind,8)
            self.assertEqual(packet_bytes(m),16+4*m.num_parameters())

    def test_zero_rule_leakage(self):
        # User data has only rule IDs, teacher may not be visible to student.
        a,b,c=generate('aligned_mirror_teacher',17,'train',384)
        x,r,y=generate('aligned_mirror_teacher',17,'train',384,'corrupted')
        self.assertTrue(torch.equal(a,x));self.assertTrue(torch.equal(b,r))
        self.assertGreater((y-c).abs().sum().item(),0)

    def test_splits_differ_and_repeat_deterministically(self):
        a=generate('aligned_mirror_teacher',17,'train',256)
        b=generate('aligned_mirror_teacher',17,'dev',256)
        c=generate('aligned_mirror_teacher',17,'train',256)
        self.assertTrue(torch.equal(a[0],c[0]))
        self.assertFalse(torch.equal(a[0],b[0]))

    def test_worlds_not_identical(self):
        x,r,_=generate('aligned_mirror_teacher',17,'train',32)
        from data import teacher_output
        self.assertGreater((teacher_output(x,r,'aligned_mirror_teacher')-teacher_output(x,r,'independent_mlp_teacher')).abs().max().item(),0.01)

    def test_forward_and_backward_finite(self):
        torch.manual_seed(9)
        x=torch.randn(16,4);r=torch.randint(0,3,(16,))
        for kind in ('shared','film','mirror','independent'):
            m=FeatureModel(kind,4)
            output=m(x,r)
            self.assertEqual(tuple(output.shape),(16,4))
            output.pow(2).mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_loop_different_depth_equal_params(self):
        torch.manual_seed(88)
        x=torch.randn(8,4)
        a=LoopModel(1);b=LoopModel(3)
        b.load_state_dict(a.state_dict())
        self.assertEqual(a.num_parameters(),b.num_parameters())
        self.assertFalse(torch.allclose(a(x),b(x)))

    def test_mirror_direction_scale_is_gauge(self):
        # Householder axis length is not an independent functional direction.
        torch.manual_seed(414)
        x=torch.randn(8,4);r=torch.tensor([0,1,2,0,1,2,0,1])
        m=FeatureModel('mirror',8)
        a=m(x,r)
        with torch.no_grad():
            m.code.mul_(3.7)
        b=m(x,r)
        self.assertLess((a-b).abs().max().item(),1e-6)

    def test_oracle_filter_uses_only_clean_examples(self):
        raw=generate('independent_mlp_teacher',17,'train',384)
        corrupt=generate('independent_mlp_teacher',17,'train',384,'corrupted')
        curated=generate('independent_mlp_teacher',17,'train',384,'oracle_filtered')
        self.assertTrue(len(curated[0])<384)
        self.assertTrue(len(curated[0])>200)
        # Every retained row must come from the same original supervised dataset.
        present={tuple(v.tolist()) for v in raw[0]}
        self.assertTrue(all(tuple(v.tolist()) in present for v in curated[0]))
        self.assertFalse(torch.equal(raw[2],corrupt[2]))

    def test_distinct_shift(self):
        a=generate('loop_teacher',17,'fresh',128)
        b=generate('loop_teacher',17,'ood',128)
        self.assertFalse(torch.allclose(a[0].mean(0),b[0].mean(0),atol=0.2))

if __name__=='__main__': unittest.main()
