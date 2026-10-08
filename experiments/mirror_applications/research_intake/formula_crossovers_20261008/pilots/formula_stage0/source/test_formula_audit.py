import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_formula_audit import algebra_checks,CHECKS,DEV,FRESH

class FormulaTests(unittest.TestCase):
    def test_each_development_seed(self):
        for seed in DEV:
            with self.subTest(seed=seed):
                checks,meta=algebra_checks(seed)
                self.assertEqual(set(checks),set(CHECKS))
                self.assertLess(max(checks.values()),1e-10)
                self.assertGreater(meta['unsafe_cache_output_difference'],1e-8)
                self.assertGreater(meta['cross_block_diag_underestimate'],1)

    def test_each_frozen_fresh_seed(self):
        for seed in FRESH:
            with self.subTest(seed=seed):
                checks,meta=algebra_checks(seed)
                self.assertEqual(set(checks),set(CHECKS))
                self.assertLess(max(checks.values()),1e-10)
                self.assertGreater(meta['unsafe_cache_output_difference'],1e-8)
                self.assertGreaterEqual(meta['coupling_majorizer_gap'],-1e-10)
                self.assertEqual(meta['joint_only_gain'],5)
                self.assertEqual(meta['k_support'],4)
                self.assertAlmostEqual(meta['reachability_Emax'],100.,places=9)

if __name__=='__main__':
    unittest.main(verbosity=2)
