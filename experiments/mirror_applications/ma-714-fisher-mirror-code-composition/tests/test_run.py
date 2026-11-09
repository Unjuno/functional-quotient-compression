import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class FisherCodeMerge(unittest.TestCase):
 def test_native_alias_and_full_fisher_reference(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('dev-71401',71401,Path(d),rhos=(0.0,))
  by={r['method']:r for r in rows}
  self.assertEqual(by['Mirror_Fisher_code']['payload_sha256'],by['native_projected_Fisher_code']['payload_sha256'])
  self.assertLessEqual(by['full_parameter_Fisher']['mean_normalized_fisher_loss'],by['Mirror_Fisher_code']['mean_normalized_fisher_loss']+1e-10)
  self.assertLess(by['Mirror_Fisher_code']['serialized_bytes'],by['full_parameter_Fisher']['serialized_bytes'])
if __name__=='__main__':unittest.main()
