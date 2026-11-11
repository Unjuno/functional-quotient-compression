import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class RoleQueryTests(unittest.TestCase):
 def test_payload_sizes_match_fixed_binary_layout(self):
  a=np.zeros((run.R,run.RANK));b=np.zeros((run.Q,run.RANK));u=np.zeros((run.R,run.Q));mask=np.ones_like(u,dtype=bool)
  self.assertEqual(len(run.payload_factor(a,b)),136);self.assertEqual(len(run.payload_flat(u,mask)),271)
 def test_observed_mask_has_unseen_role_query_pairs(self):
  u,m=run.make_world(11);self.assertGreater(m.sum(),0);self.assertGreater((~m).sum(),0);self.assertEqual(u.shape,(run.R,run.Q))
 def test_factorized_decoder_uses_two_messages_per_query(self):
  self.assertEqual(run.Q*run.K,16);self.assertEqual(run.Q*run.R,64);self.assertLess(run.Q*run.K,run.Q*run.R)
if __name__=='__main__':unittest.main()
