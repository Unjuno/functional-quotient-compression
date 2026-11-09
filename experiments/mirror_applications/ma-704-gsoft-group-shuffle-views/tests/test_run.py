import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class GSOFTViews(unittest.TestCase):
 def test_exact_transform_storage_and_runtime_proxy(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-70411',70411,Path(d));by={r['shuffle_stages']:r for r in rows}
   for r in rows:
    self.assertLessEqual(r['max_output_nrmse'],1e-5)
    self.assertEqual(r['payload_sha256'],r['native_sha256'])
   self.assertLess(by[2]['serialized_bytes']/by[2]['independent_bytes'],0.5)
   self.assertLess(by[2]['active_compute_proxy']/by[2]['independent_compute_proxy'],1.3)
   self.assertGreater(by[2]['structured_wall_s'],by[2]['dense_wall_s'])
   p=Path(d)/'audit-70411_s2.gsoft.npz'
   self.assertEqual(p.stat().st_size,by[2]['serialized_bytes'])
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),by[2]['payload_sha256'])
if __name__=='__main__':unittest.main()
