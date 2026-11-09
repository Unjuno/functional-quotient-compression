import tempfile,sys,unittest,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class AtomPruning(unittest.TestCase):
 def test_development_failure_and_control_replay(self):
  with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
   warnings.simplefilter('ignore')
   rows=run.run_one('dev-71001',71001,Path(d),rhos=(0.0,))
  by={r['method']:r for r in rows}
  self.assertEqual(by['mirror_sparse_fullbank']['payload_sha256'],by['native_sparse_coding']['payload_sha256'])
  self.assertGreater(by['postfit_pruned']['mean_heldout_nrmse'],1e-5)
  self.assertGreater(by['postfit_pruned']['serialized_bytes'],by['independent_toeplitz']['serialized_bytes'])
  self.assertLess(by['random_pruned']['mean_heldout_nrmse'],by['postfit_pruned']['mean_heldout_nrmse'])
  self.assertLess(by['independent_toeplitz']['mean_heldout_nrmse'],1e-5)
  self.assertLess(by['fullbank_dense_codes']['mean_heldout_nrmse'],1e-5)
if __name__=='__main__':unittest.main()
