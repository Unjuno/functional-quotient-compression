import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class FNOViews(unittest.TestCase):
 def test_determinism_alias_and_resolution(self):
  a=run.run(73101);b=run.run(73101)
  self.assertEqual([(x['payload_sha256'],x['relative_solution_error']) for x in a],[(x['payload_sha256'],x['relative_solution_error']) for x in b])
  d={(x['method'],x['basis_rank'],x['resolution']):x for x in a}
  for q in run.RANKS:
   for r in run.RES:self.assertEqual(d['Mirror_operator_codes',q,r]['payload_sha256'],d['native_PCA_operator_codes',q,r]['payload_sha256'])
  self.assertEqual(len(a),2*(4+4*2))
if __name__=='__main__':unittest.main()
