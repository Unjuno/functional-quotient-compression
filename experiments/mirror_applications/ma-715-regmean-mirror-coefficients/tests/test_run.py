import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import run

class RegMeanMirror(unittest.TestCase):
    def test_complete_payload_and_native_alias(self):
        rows=run.run(seed=71501)
        by={r['method']:r for r in rows if r['request']==0}
        self.assertEqual(by['Mirror_basis']['payload_sha256'],by['native_projected_RegMean']['payload_sha256'])
        self.assertGreater(by['Mirror_basis']['serialized_bytes'],0)
        self.assertGreater(by['full_RegMean']['serialized_bytes'],0)
        self.assertEqual(len(rows),48)
        self.assertEqual(by['full_RegMean']['normalized_excess_objective'],0.0)
    def test_deterministic_replay(self):
        a=run.run(seed=71501); b=run.run(seed=71501)
        self.assertEqual([(x['payload_sha256'],x['nrmse_to_full_RegMean']) for x in a],[(x['payload_sha256'],x['nrmse_to_full_RegMean']) for x in b])
if __name__=='__main__': unittest.main()
