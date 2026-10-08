import csv
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from run_gauge import world,rotation,tangent_delta

class GaugeStage0Tests(unittest.TestCase):
    def test_generic_pre_rope_rank_and_kernel(self):
        x=world(9)
        self.assertEqual((x['rank_no_rope'],x['nullity_no_rope']),(12,4))
        self.assertLess(x['full_no_rope_gauge_tangent_inf'],1e-10)

    def test_rope_reduces_symmetry_rank(self):
        x=world(9)
        self.assertEqual((x['rank_rope'],x['nullity_rope']),(14,2))
        self.assertLess(x['valid_rope_gauge_tangent_inf'],1e-10)

    def test_invalid_rope_transform_is_detected(self):
        self.assertGreater(world(9)['invalid_rope_exact_error'],1e-3)

    def test_all_fresh_worlds_are_recorded(self):
        with (pathlib.Path(__file__).parent/'fresh_raw.csv').open(newline='') as f:
            x=list(csv.DictReader(f))
        self.assertEqual([int(r['seed']) for r in x],[101,102,103,104,105])
        self.assertTrue(all((int(r['rank_no_rope']),int(r['rank_rope']))==(12,14) for r in x))

if __name__=='__main__': unittest.main(verbosity=2)
