import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class ToeplitzMirror(unittest.TestCase):
 def test_model_payload_replay_and_native_alias(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-70911',70911,Path(d));by={(x['private_kernel_fraction'],x['method']):x for x in rows}
   for rho in run.RHOS:
    a=by[(rho,'mirror_shared_shift_gain')];b=by[(rho,'native_shared_shift_gain')]
    self.assertEqual(a['payload_sha256'],b['payload_sha256']);self.assertEqual(a['serialized_bytes'],b['serialized_bytes'])
    self.assertLess(by[(rho,'independent_toeplitz')]['mean_heldout_nrmse'],1e-5)
    self.assertLess(by[(rho,'independent_dense')]['mean_heldout_nrmse'],1e-5)
   self.assertLess(by[(0.0,'mirror_shared_shift_gain')]['mean_heldout_nrmse'],1e-5)
   self.assertGreater(by[(0.2,'mirror_shared_shift_gain')]['mean_heldout_nrmse'],0.05)
   self.assertLess(by[(0.0,'mirror_shared_shift_gain')]['serialized_bytes'],by[(0.0,'independent_toeplitz')]['serialized_bytes'])
if __name__=='__main__':unittest.main()
