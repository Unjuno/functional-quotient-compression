import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class LogicalExperts(unittest.TestCase):
 def test_householder_and_film_controls(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-70211',70211,Path(d));by={r['reflection_count']:r for r in rows}
   for r in rows:
    self.assertLess(r['max_householder_output_nrmse'],1e-5)
    self.assertEqual(r['payload_sha256'],r['native_sha256'])
   self.assertLess(by[4]['serialized_bytes']/by[4]['independent_bytes'],0.25)
   self.assertLess(by[4]['active_compute_proxy']/by[4]['independent_compute_proxy'],1.1)
   self.assertGreater(by[1]['film_output_nrmse'],0.02)
   p=Path(d)/'audit-70211_r4.householder.npz'
   self.assertEqual(p.stat().st_size,by[4]['serialized_bytes'])
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),by[4]['payload_sha256'])
if __name__=='__main__':unittest.main()
