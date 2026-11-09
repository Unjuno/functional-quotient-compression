import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class DELLAView(unittest.TestCase):
 def test_native_pca_alias_and_sparse_payload(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('dev-71301',71301,Path(d),rhos=(0.0,))
  by={x['method']:x for x in rows}
  for r in (1,2,4,8):self.assertEqual(by[f'Magnitude_PCA_r{r}']['payload_sha256'],by[f'native_PCA_r{r}']['payload_sha256'])
  self.assertGreater(by['magnitude_topk']['serialized_bytes'],0)
  self.assertLess(by['magnitude_topk']['serialized_bytes'],by['independent_dense']['serialized_bytes'])
if __name__=='__main__':unittest.main()
