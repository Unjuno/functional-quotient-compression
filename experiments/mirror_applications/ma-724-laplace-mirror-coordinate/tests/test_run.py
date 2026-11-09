import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class LaplaceCodes(unittest.TestCase):
 def test_alias_and_replay(self):
  a=run.run(72401);b=run.run(72401)
  self.assertEqual([(x['payload_sha256'],x['nll']) for x in a],[(x['payload_sha256'],x['nll']) for x in b])
  d={(x['method'],x['rank']):x for x in a}
  for q in run.QS:self.assertEqual(d['native_subspace_Laplace',q]['payload_sha256'],d['Mirror_code_Laplace',q]['payload_sha256'])
  self.assertEqual(len(a),11)
if __name__=='__main__':unittest.main()
