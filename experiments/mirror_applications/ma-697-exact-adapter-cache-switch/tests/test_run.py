import tempfile,unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class AdapterCacheSwitch(unittest.TestCase):
 def test_exact_switch_and_cost_frontier(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-69711',69711,Path(d))
   r=rows[0]
   self.assertLessEqual(r['max_lazy_rel_error'],1e-5)
   self.assertLessEqual(r['serialized_bytes']/r['independent_bytes'],0.25)
   self.assertEqual(r['transform_only_break_even_tokens'],128)
   self.assertGreater(r['direct_cache_reuse_max_rel_error'],0.01)
   self.assertEqual(r['native_rotation_rel_error'],0.0)
   p=Path(d)/'audit-69711.shared.npz'
   self.assertEqual(p.stat().st_size,r['serialized_bytes'])
   self.assertEqual(r['shared_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
