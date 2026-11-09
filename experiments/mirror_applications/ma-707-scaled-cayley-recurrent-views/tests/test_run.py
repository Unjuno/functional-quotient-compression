import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class CayleyRecurrence(unittest.TestCase):
 def test_stable_serialized_replay_and_cost(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-70711',70711,Path(d));by={r['skew_rank']:r for r in rows}
   for r in rows:
    self.assertLessEqual(r['trajectory_nrmse'],1e-5)
    self.assertLess(r['max_spectral_radius'],1.0)
    self.assertEqual(r['payload_sha256'],r['native_sha256'])
   self.assertLess(by[1]['serialized_bytes']/by[1]['independent_bytes'],0.4)
   self.assertGreater(by[1]['reconstruction_overhead_fraction'],0.25)
   p=Path(d)/'audit-70711_r1.mirror.npz'
   self.assertEqual(p.stat().st_size,by[1]['serialized_bytes'])
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),by[1]['payload_sha256'])
if __name__=='__main__':unittest.main()
