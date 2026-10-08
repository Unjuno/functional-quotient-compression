import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model import MODES,REPS,Stitcher,make_split,givens

class StitchingTests(unittest.TestCase):
    def test_givens_is_orthogonal(self):
        r=givens(torch.tensor(.37));self.assertTrue(torch.allclose(r.T@r,torch.eye(2),atol=1e-6))

    def test_representation_information_boundary(self):
        reps,y=make_split(93200,'dev',2048)
        full_probe=torch.linalg.lstsq(reps['full'],y).solution
        partial_probe=torch.linalg.lstsq(reps['task0_only'],y).solution
        noise_probe=torch.linalg.lstsq(reps['unrelated_noise'],y).solution
        self.assertLess(((reps['full']@full_probe-y)**2).mean(0)[1].item(),.01)
        self.assertGreater(((reps['task0_only']@partial_probe-y)**2).mean(0)[1].item(),.5)
        self.assertGreater(((reps['unrelated_noise']@noise_probe-y)**2).mean(0).mean().item(),.7)

    def test_all_stitchers_forward_backward(self):
        reps,y=make_split(93201,'train',32);x=reps['full'];task=torch.randint(2,(32,))
        for mode in MODES:
            model=Stitcher(mode,17);out=model(x,task);self.assertEqual(out.shape,(32,))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
