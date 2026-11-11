import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'source'/'run.py'; spec=importlib.util.spec_from_file_location('runner',p); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class TestFactorScreen(unittest.TestCase):
 def test_compositions_recover_and_reset(self):
  r=runner.run(37)
  self.assertEqual(r['heldout_pairs'],24)
  self.assertEqual(r['max_heldout_nrmse'],0.)
  self.assertEqual(r['reset_max_abs'],0.)
  self.assertEqual(r['fast_state_bytes'],runner.D*runner.D*4)
 def test_native_control_matches_mirror_cost(self):
  r=runner.run(59)
  self.assertEqual(r['native_query_macs'],r['mirror_query_macs'])
  self.assertLess(r['native_factor_bytes'],r['mirror_bytes'])
if __name__=='__main__': unittest.main()
