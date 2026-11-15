import sys,unittest
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'source'))
from adapters import Bottleneck,MirrorGivens,SharedAdapter,FiLMControl,MoPEFTLike,IndependentAdapters

class SAMFeatureAdapterTests(unittest.TestCase):
    def test_all_feature_peft_controls_preserve_sam_embedding_shape(self):
        x=torch.randn(1,256,8,8)
        for cls in (MirrorGivens,SharedAdapter,FiLMControl,MoPEFTLike,IndependentAdapters):
            y=cls().apply(x,1)
            self.assertEqual(tuple(y.shape),tuple(x.shape))
    def test_zero_givens_is_shared_adapter_function(self):
        torch.manual_seed(8);shared=SharedAdapter();mirror=MirrorGivens()
        mirror.adapter.load_state_dict(shared.adapter.state_dict());mirror.angles.data.zero_();x=torch.randn(1,256,8,8)
        torch.testing.assert_close(mirror.apply(x,0),shared.apply(x,0),rtol=0,atol=0)
    def test_zero_film_is_shared_adapter_function(self):
        torch.manual_seed(9);shared=SharedAdapter();film=FiLMControl()
        film.adapter.load_state_dict(shared.adapter.state_dict());film.scale.data.zero_();film.shift.data.zero_();x=torch.randn(1,256,8,8)
        torch.testing.assert_close(film.apply(x,2),shared.apply(x,2),rtol=0,atol=0)
    def test_mopeft_gates_are_normalized(self):
        m=MoPEFTLike();x=torch.randn(1,256,8,8);y=m.apply(x,0)
        self.assertEqual(tuple(y.shape),tuple(x.shape));self.assertAlmostEqual(float(torch.softmax(m.gate_logits[0].detach(),-1).sum()),1.0,places=7)
    def test_registered_code_coordinates_are_small(self):
        m=MirrorGivens();self.assertEqual(m.angles.numel(),6)
        self.assertLess(m.angles.numel()*m.angles.element_size(),m.adapter.down.weight.numel()*m.adapter.down.weight.element_size())
if __name__=='__main__':unittest.main()
