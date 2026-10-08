import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import METHODS,BeamCodebook,make_channels,rates

class BeamTests(unittest.TestCase):
    def test_phase_only_unit_norm_and_code_shapes(self):
        for i,method in enumerate(METHODS):
            model=BeamCodebook(method,98100+i)
            for sector in range(4):
                beam=model.beams(sector)
                self.assertEqual(tuple(beam.shape),(8,8))
                self.assertTrue(torch.allclose(beam.abs(),torch.full_like(beam.abs(),1/(8**.5)),atol=1e-6))

    def test_channel_shapes_and_finite_rate(self):
        h=make_channels(98101,'development',32)
        self.assertEqual(tuple(h.shape),(4,32,8))
        self.assertTrue(torch.isfinite(h.real).all() and torch.isfinite(h.imag).all())
        m=BeamCodebook('mirror_view',98101);r=rates(h[0],m.beams(0))
        self.assertEqual(tuple(r.shape),(32,8));self.assertTrue(torch.isfinite(r).all())

    def test_mirror_parameters_receive_gradient(self):
        h=make_channels(98102,'train',16);m=BeamCodebook('mirror_view',98102)
        r=rates(h[0],m.beams(0));loss=-.2*torch.logsumexp(r/.2,dim=1).mean();loss.backward()
        self.assertTrue(torch.isfinite(m.base.grad).all());self.assertTrue(torch.isfinite(m.offset.grad).all())

if __name__=='__main__':unittest.main()
