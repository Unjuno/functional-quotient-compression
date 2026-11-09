import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class GSOFTExperts(unittest.TestCase):
 def test_expert_quality_bytes_controls_and_runtime(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.main_one('audit-70511',70511,Path(d));by={r['shuffle_stages']:r for r in rows}
   for r in rows:
    self.assertLess(r['max_gsoft_output_nrmse'],1e-5)
    self.assertEqual(r['payload_sha256'],r['native_sha256'])
    self.assertGreater(r['wall_time_s'],0)
   self.assertLess(by[2]['serialized_bytes']/by[2]['independent_bytes'],0.25)
   self.assertLess(by[2]['active_compute_proxy']/by[2]['independent_compute_proxy'],1.2)
   self.assertGreater(by[1]['ia3_output_nrmse'],0.02)
   self.assertGreater(by[2]['rank1_output_nrmse'],0.02)
   p=Path(d)/'audit-70511_s2.gsoft.npz'
   self.assertEqual(p.stat().st_size,by[2]['serialized_bytes'])
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),by[2]['payload_sha256'])
if __name__=='__main__':unittest.main()
