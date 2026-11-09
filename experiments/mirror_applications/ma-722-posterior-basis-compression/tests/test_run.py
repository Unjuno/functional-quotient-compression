import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class BasisCompression(unittest.TestCase):
 def test_alias_and_replay(self):
  a=run.run(72201);b=run.run(72201)
  self.assertEqual([(x['payload_sha256'],x['predictive_nll']) for x in a],[(x['payload_sha256'],x['predictive_nll']) for x in b])
  d={(x['method'],x['factor_rank']):x for x in a}
  for q in run.QS:self.assertEqual(d['native_SVD_factor',q]['payload_sha256'],d['Mirror_factor_code',q]['payload_sha256'])
  self.assertEqual(len(a),12)
if __name__=='__main__':unittest.main()
