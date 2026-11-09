import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run
class ParameterSuperposition(unittest.TestCase):
 def test_hadamard_is_native_alias_and_counts_exact(self):
  with tempfile.TemporaryDirectory() as d:
   rows=run.run_one('dev-71201',71201,Path(d),counts=(8,32),rhos=(0.0,))
  by={(r['task_count'],r['method']):r for r in rows}
  for n in (8,32):
   a=by[(n,'Mirror_Hadamard_DARE')];b=by[(n,'native_PS_Hadamard_DARE')]
   self.assertEqual(a['payload_sha256'],b['payload_sha256'])
   self.assertEqual(a['serialized_bytes'],b['serialized_bytes'])
   self.assertGreater(by[(n,'independent_DARE')]['dare_nonzeros'],0)
   self.assertLess(by[(n,'independent_dense')]['serialized_bytes'],10**7)
if __name__=='__main__':unittest.main()
