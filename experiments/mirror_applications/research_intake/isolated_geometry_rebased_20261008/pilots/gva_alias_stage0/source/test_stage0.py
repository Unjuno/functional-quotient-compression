import csv
import math
import pathlib
import unittest
import torch
import sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from run_stage0 import gva_forward,world

class Stage0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)

    def test_query_absorption_identity(self):
        a=world(9)
        self.assertLess(a['score_error'],3e-5)
        self.assertLess(a['output_error'],3e-5)

    def test_native_factorized_control_cannot_be_outsaved(self):
        a=world(9)
        self.assertEqual(a['native_linear_code_npz_bytes'],a['mirror_maps_npz_bytes'])

    def test_real_cache_alias_and_unsafe_counterexample(self):
        a=world(9)
        self.assertTrue(a['cache_alias'])
        self.assertEqual(a['cache_bytes_cloned'], 8*a['cache_bytes'])
        self.assertGreater(a['unsafe_output_error'],1e-4)

    def test_rope_valid_invalid(self):
        a=world(9)
        self.assertLess(a['rope_valid_commutator_error'],1e-12)
        self.assertGreater(a['rope_invalid_commutator_error'],1e-4)

    def test_all_five_fresh_records(self):
        csv_path=pathlib.Path(__file__).parent/'fresh_raw.csv'
        with csv_path.open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual([int(r['seed']) for r in rows],[101,102,103,104,105])
        self.assertTrue(all(int(r['cache_alias'])==1 for r in rows))

if __name__=='__main__': unittest.main(verbosity=2)
