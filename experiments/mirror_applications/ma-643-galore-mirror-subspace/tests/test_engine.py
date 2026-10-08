import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import engine

class ScreenTests(unittest.TestCase):
    def test_payload_roundtrip_deterministic(self):
        b,i,c,_=engine.world(64301)
        blob,meta,arr=engine.payload('mirror_sparse_atom_address',b,i,c)
        self.assertEqual(len(blob),len(engine.payload('mirror_sparse_atom_address',b,i,c)[0]))
        self.assertEqual(meta['method'],'mirror_sparse_atom_address')
        self.assertEqual(arr['indices'].dtype,np.uint8)

    def test_mirror_and_ordinary_codes_reconstruct_same_direction(self):
        b,i,c,_=engine.world(64301)
        _,_,m=engine.payload('mirror_sparse_atom_address',b,i,c)
        _,_,o=engine.payload('ordinary_dense_atom_coefficients',b,i,c)
        np.testing.assert_allclose(engine.reconstruct('mirror_sparse_atom_address',m),engine.reconstruct('ordinary_dense_atom_coefficients',o),atol=1e-6)

    def test_hard_tie_cannot_fit_nonzero_task(self):
        r=engine.evaluate(64301,'hard_shared_atom_tie')
        self.assertAlmostEqual(r['final_objective_gap'],1.0)

    def test_projected_and_full_gradient_reach_target(self):
        for method in ('galore_projected_adam_state','mirror_sparse_atom_address','ordinary_dense_atom_coefficients','full_gradient_upper_control'):
            self.assertLess(engine.evaluate(64301,method)['final_objective_gap'],1e-3)

if __name__=='__main__': unittest.main()
