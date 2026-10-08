import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import engine

class FactorizedFastMemoryTests(unittest.TestCase):
    def test_hadamard_and_mirror_match_all_pairings(self):
        m=engine.evaluate(85301,'mirror_factorized_role_code')
        v=engine.evaluate(85301,'vsa_hadamard_binding')
        self.assertAlmostEqual(m['normalized_heldout_mse'],v['normalized_heldout_mse'],places=12)
        self.assertEqual(m['heldout_exact_nearest_accuracy'],1.0)
        self.assertLess(abs(m['serialized_bytes']-v['serialized_bytes']),16)
    def test_delta_state_recalls_written_pairs_only(self):
        _,arr,_=engine.state('delta_rule_fast_matrix',85301)
        for pair in engine.TRAIN:
            self.assertLess(float(abs(engine.predict('delta_rule_fast_matrix',arr,pair)-engine.pair_values(*engine.world(85301))[engine.PAIRS.index(pair)]).max()),1e-6)
        self.assertEqual(float(abs(engine.predict('delta_rule_fast_matrix',arr,(0,0))).max()),0.0)
    def test_typed_payload_roundtrip(self):
        meta,arr,_=engine.state('mirror_factorized_role_code',85301);blob=engine.serialize(meta,arr);m2,a2=engine.deserialize(blob)
        self.assertEqual(engine.serialize(m2,a2),blob)
        self.assertEqual(a2['pair_ids'].dtype,arr['pair_ids'].dtype)

if __name__=='__main__':unittest.main()
