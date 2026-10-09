import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

class Mechanism(unittest.TestCase):
    def test_exact_heldout_and_distinct_paths(self):
        for seed in (36601,36602):
            w=run.world(seed); s=run.states(w)['mirror_path'][0]
            err,distinct,route=run.score('mirror_path',s,w)
            self.assertLessEqual(err,1e-6); self.assertEqual(distinct,16); self.assertEqual(route,2)
    def test_direct_control_same_function(self):
        w=run.world(36601); a=run.states(w)
        for i,j in w[7]:
            self.assertTrue(np.array_equal(run.decode('mirror_path',a['mirror_path'][0],i,j),run.decode('direct_coefficients',a['direct_coefficients'][0],i,j)))
    def test_actual_payload_and_roundtrip(self):
        w=run.world(36601); states=run.states(w)
        for name in ('flat_paths','pa02_factorized','mirror_path','direct_coefficients'):
            payload,decoded,meta=run.pack(*states[name])
            self.assertGreater(len(payload),0); self.assertEqual(meta['kind'],states[name][1]['kind'])
            for k,v in states[name][0].items(): self.assertTrue(np.array_equal(v,decoded[k]))

if __name__=='__main__': unittest.main()
