import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class AlignmentTests(unittest.TestCase):
 def test_alignment_permutation_preserves_model_logits(self):
  torch.manual_seed(11);ref=run.Net();model=run.Net();probe=torch.randn(40,64);_,diff,_=run.align_to_reference(ref,model,probe);self.assertLessEqual(diff,1e-6)
 def test_activation_alignment_is_a_bijection(self):
  torch.manual_seed(12);ref=run.Net();model=run.Net();p,_,_=run.align_to_reference(ref,model,torch.randn(40,64));self.assertEqual(sorted(p.tolist()),list(range(run.WIDTH)))
 def test_mirror_coordinate_roundtrip_exact_grid(self):
  for uv in run.COORDS:
   m=2*uv-1;d=(m+1)/2;self.assertTrue((d==uv).all())
if __name__=='__main__':unittest.main()
