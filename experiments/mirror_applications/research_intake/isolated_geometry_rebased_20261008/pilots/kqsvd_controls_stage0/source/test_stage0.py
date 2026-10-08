import unittest
import numpy as np
from run_stage0 import one_world,build_world,query,native_kq_projector,key_projector,relscore

class Tests(unittest.TestCase):
    def test_full_rank(self):
        rng,K,V=build_world(11)
        Q=query(rng,V,0,96)
        self.assertEqual(np.linalg.matrix_rank(K),16)
        self.assertEqual(np.linalg.matrix_rank(Q),16)

    def test_native_calibration_exact_rank4(self):
        rng,K,V=build_world(13)
        Q=query(rng,V,3,96)
        P,error=native_kq_projector(Q,K)
        self.assertLess(error,1e-9)
        self.assertEqual(np.linalg.matrix_rank(P),4)

    def test_score_svd_oracle_bound(self):
        for seed in (11,12,13,101,102,103,104,105):
            d=one_world(seed,'test')
            self.assertLessEqual(d['oracle_rank4_rel_score'],d['key_svd_rel_score']+1e-10)
            self.assertTrue(d['ordinary_linear_code_equals_proposed_linear_m'])

    def test_physical_byte_counts(self):
        d=one_world(101,'test')
        self.assertLess(d['physical_key_bytes_npz'],d['linear_code_bank_bytes_npz'])
        self.assertLess(d['physical_key_bytes_npz'],d['native_full_bank_bytes_npz'])

if __name__=='__main__':
    unittest.main(verbosity=2)
