import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class YOCOViews(unittest.TestCase):
 def test_layer_view_algebra_frontier(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-69511',69511,Path(d))
   by={(r['rho'],r['private_rank']):r for r in rows}
   aligned=by[(0.0,0)]
   self.assertLess(aligned['max_layer_output_nrmse'],1e-5)
   self.assertLess(aligned['serialized_bytes']/aligned['independent_bytes'],0.75)
   self.assertLess(aligned['active_compute_proxy']/aligned['independent_compute_proxy'],2.0)
   self.assertEqual(aligned['payload_sha256'],aligned['native_control_sha256'])
   self.assertEqual(aligned['native_basis_alias_nrmse'],0.0)
   mild=by[(0.1,1)]
   self.assertGreater(mild['max_layer_output_nrmse'],0.02)
   exact=by[(0.1,4)]
   self.assertLess(exact['max_layer_output_nrmse'],1e-5)
   self.assertLess(exact['serialized_bytes']/exact['independent_bytes'],0.75)
if __name__=='__main__':unittest.main()
