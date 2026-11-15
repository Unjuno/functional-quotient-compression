import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("ma1035", SRC / "run_experiment.py")
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)


class QINCoMirrorTests(unittest.TestCase):
    def test_packed_indices_roundtrip_exact(self):
        codes = np.arange(40, dtype=np.uint8).reshape(10, 4) % 16
        self.assertTrue(np.array_equal(codes, ma.unpack_codes(ma.pack_codes(codes), len(codes))))
        self.assertEqual(len(ma.pack_codes(codes)), len(codes) * 2)

    def test_givens_is_norm_preserving_and_changes_condition(self):
        x = torch.randn(5, ma.D)
        y = ma.givens_view(x, torch.tensor(0.37))
        self.assertTrue(torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-6))
        self.assertFalse(torch.allclose(x, y))

    def test_shared_variants_encode_decode(self):
        torch.manual_seed(9)
        data = torch.randn(12, ma.D)
        for method in ["qinco_tied", "qinco_mirror_givens", "qinco_additive_rank1"]:
            model = ma.SharedStageQINCo(method)
            codes, recon = model.encode(data)
            decoded = model.decode(codes)
            self.assertEqual(tuple(codes.shape), (12, ma.M))
            self.assertTrue(torch.equal(recon, decoded))
            self.assertTrue(torch.isfinite(decoded).all())

    def test_official_native_qinco_step_count(self):
        from qinco_reference import QINCo
        model = QINCo(ma.D, ma.K, ma.L, ma.M, ma.H)
        self.assertEqual(len(model.steps), ma.M - 1)
        self.assertEqual(model.steps[0].codebook.weight.shape, (ma.K, ma.D))


if __name__ == "__main__":
    unittest.main()
