import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from run import METHODS, QRModel, pair_split, make_data, deterministic_pack, model_arrays, load_serialized


class QRTests(unittest.TestCase):
    def test_pair_split_is_exact_quarter_and_each_partition_seen(self):
        held = pair_split(39101).reshape(12, 12)
        self.assertEqual(int(held.sum()), 36)
        self.assertTrue((~held).any(axis=0).all())
        self.assertTrue((~held).any(axis=1).all())

    def test_data_reproducible_and_held_pairs_excluded_from_train(self):
        a, b = make_data(39101), make_data(39101)
        self.assertTrue(torch.equal(a["train"]["tokens"], b["train"]["tokens"]))
        self.assertFalse(a["train"]["held"].any())
        self.assertTrue(a["validation"]["held"].any())

    def test_payload_deterministic_and_reloads(self):
        model = QRModel("mirror_factorized", 39101)
        arrays = model_arrays(model)
        with self.subTest("serialization"):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                a, ha = deterministic_pack(arrays, Path(directory) / "a.npz")
                b, hb = deterministic_pack(arrays, Path(directory) / "b.npz")
                self.assertEqual((a, ha), (b, hb))
        restored = load_serialized("mirror_factorized", 39101, arrays)
        tokens = torch.arange(144)
        self.assertTrue(torch.allclose(model(tokens), restored(tokens), atol=2e-4, rtol=2e-4))


if __name__ == "__main__":
    unittest.main()
