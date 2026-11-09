import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class DAREMirror(unittest.TestCase):
 def test_payload_alias_and_sparse_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('dev-71101',71101,Path(d),rhos=(0.0,))
  by={r['method']:r for r in rows}
  for rank in (1,2,4,8):
   self.assertEqual(by[f'DARE_PCA_r{rank}']['payload_sha256'],by[f'native_PCA_r{rank}']['payload_sha256'])
   self.assertEqual(by[f'DARE_PCA_r{rank}']['serialized_bytes'],by[f'native_PCA_r{rank}']['serialized_bytes'])
  self.assertLess(by['DARE_sparse']['serialized_bytes'],by['independent_dense']['serialized_bytes'])
  self.assertGreater(by['DARE_sparse']['dare_nonzeros'],0)
  self.assertGreater(by['task_arithmetic_mean']['merge_interference_nrmse'],0.5)
if __name__=='__main__':unittest.main()
