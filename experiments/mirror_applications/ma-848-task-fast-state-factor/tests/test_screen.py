import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'source'/'run.py'; spec=importlib.util.spec_from_file_location('runner',p); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class TestFactorScreen(unittest.TestCase):
 def test_compositions_recover_and_reset(self):
  r=runner.run(37)
  self.assertEqual(r['heldout_pairs'],24)
  self.assertLess(r['max_heldout_nrmse'],1e-6)
  self.assertEqual(r['reset_max_abs'],0.)
 def test_native_control_dominates_mirror_cost(self):
  r=runner.run(59)
  self.assertAlmostEqual(r['mean_heldout_nrmse'],2.495161959049289e-8,places=12)
  self.assertLess(r['native_factor_bytes'],r['mirror_bytes'])
if __name__=='__main__': unittest.main()
