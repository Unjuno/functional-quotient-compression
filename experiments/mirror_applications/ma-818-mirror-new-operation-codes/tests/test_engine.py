import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import engine


class OperationCodeTests(unittest.TestCase):
    def test_mirror_is_identical_to_ordinary_structured_embedding(self):
        for seed in (81801, 81802):
            for support in engine.SUPPORTS:
                for relation in ('aligned', 'unrelated'):
                    mirror = engine.evaluate(seed, 'structured_mirror_angles', support, relation)
                    ordinary = engine.evaluate(seed, 'ordinary_structured_operation_embedding', support, relation)
                    self.assertEqual(mirror['serialized_bytes'], ordinary['serialized_bytes'])
                    self.assertEqual(mirror['payload_sha256'], ordinary['payload_sha256'])
                    self.assertEqual(mirror['normalized_sequence_mse'], ordinary['normalized_sequence_mse'])

    def test_aligned_givens_operation_fit(self):
        result = engine.evaluate(81801, 'structured_mirror_angles', 8, 'aligned')
        self.assertLessEqual(result['normalized_sequence_mse'], 1e-5)
        self.assertLessEqual(result['old_operation_retention_mse'], 1e-5)

    def test_unrelated_operation_requires_private_operator(self):
        compact = engine.evaluate(81801, 'structured_mirror_angles', 16, 'unrelated')
        full = engine.evaluate(81801, 'independent_full_operator', 16, 'unrelated')
        self.assertGreater(compact['normalized_sequence_mse'], 0.1)
        self.assertLessEqual(full['normalized_sequence_mse'], 1e-5)
        self.assertGreater(full['serialized_bytes'], compact['serialized_bytes'])

    def test_old_operations_remain_available(self):
        old, _, _ = engine.world(81801, 'aligned')
        x = np.arange(engine.D, dtype=float)
        for i in range(engine.OLD):
            np.testing.assert_allclose(engine.givens(old[i]) @ x, engine.givens(old[i]) @ x)


if __name__ == '__main__':
    unittest.main()
