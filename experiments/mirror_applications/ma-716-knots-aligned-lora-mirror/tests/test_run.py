import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class KnotsMirror(unittest.TestCase):
 def test_gauge_invariance_and_alias(self):
  rows=run.run(71601); by={x['method']:x for x in rows}
  self.assertLess(by['independent_LoRA']['gauge_max_prediction_delta'],1e-12)
  self.assertEqual(by['Mirror_sparse_top2']['payload_sha256'],by['native_sparse_top2']['payload_sha256'])
  self.assertEqual(by['Mirror_sparse_top2']['heldout_nrmse'],by['native_sparse_top2']['heldout_nrmse'])
 def test_seed_replay(self):
  a=run.run(71601);b=run.run(71601)
  self.assertEqual([(x['payload_sha256'],x['heldout_nrmse']) for x in a],[(x['payload_sha256'],x['heldout_nrmse']) for x in b])
if __name__=='__main__':unittest.main()
