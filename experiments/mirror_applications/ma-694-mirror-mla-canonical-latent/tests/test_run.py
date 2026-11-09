import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class MLALatentFrontier(unittest.TestCase):
 def test_payload_frontier_and_native_alias(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('audit-69411',69411,Path(d))
   by={(r['rho'],r['private_rank']):r for r in rows}
   aligned=by[(0.0,0)]
   self.assertLessEqual(aligned['max_view_output_nrmse'],0.02)
   self.assertLessEqual(aligned['serialized_bytes']/aligned['independent_bytes'],0.75)
   self.assertEqual(aligned['payload_sha256'],aligned['native_control_sha256'])
   self.assertEqual(aligned['native_basis_alias_nrmse'],0.0)
   het=by[(0.1,2)]
   self.assertGreater(het['max_view_output_nrmse'],0.02)
   exact=by[(0.1,8)]
   self.assertLessEqual(exact['max_view_output_nrmse'],0.02)
   self.assertGreater(exact['serialized_bytes'],exact['independent_bytes'])
   p=Path(d)/'audit-69411_rho0.00_q0.mirror.npz'
   self.assertEqual(p.stat().st_size,aligned['serialized_bytes'])
if __name__=='__main__':unittest.main()
