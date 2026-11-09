import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class TopKCacheViews(unittest.TestCase):
 def test_outputs_storage_and_k_compute(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-69911',69911,Path(d));by={r['top_k']:r for r in rows}
   self.assertTrue(all(r['max_topk_output_rel_error']<=1e-5 for r in rows))
   self.assertTrue(all(r['native_phase_rel_error']==0 for r in rows))
   self.assertLess(by[8]['serialized_bytes']/by[8]['independent_bytes'],0.25)
   self.assertAlmostEqual(by[8]['compute_multiple_of_single_attention'],8.25)
   p=Path(d)/'audit-69911_k8.shared.npz'
   self.assertEqual(p.stat().st_size,by[8]['serialized_bytes'])
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),by[8]['shared_sha256'])
if __name__=='__main__':unittest.main()
