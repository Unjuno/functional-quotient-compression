import importlib.util
import pathlib
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ma1033", ROOT / "source" / "run_experiment.py")
ma = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ma)


class MechanismTests(unittest.TestCase):
    def test_product_view_has_cartesian_address(self):
        init = torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512)
        net = ma.PromptState(init, "mirror_product", 103300)
        with torch.no_grad():
            net.class_factor.copy_(torch.tensor([0.2] * len(ma.CLASSES)))
            net.domain_factor.copy_(torch.tensor([0.3, 0.4, 0.5, 0.6]))
        a = net.class_contexts(0)[0]
        b = net.class_contexts(1)[0]
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(tuple(net.class_contexts(0).shape), (len(ma.CLASSES), 2, 512))

    def test_additive_is_separate_native_control(self):
        init = torch.zeros(2, 512)
        product = ma.PromptState(init, "mirror_product", 7)
        additive = ma.PromptState(init, "additive_code", 7)
        self.assertEqual(product.class_factor.numel() + product.domain_factor.numel(),
                         additive.class_factor.numel() + additive.domain_factor.numel())
        self.assertEqual(tuple(additive.class_contexts(0).shape), (10, 2, 512))

    def test_independent_bank_and_view_payload_roundtrip_bytes(self):
        init = torch.zeros(2, 512)
        small = ma.PromptState(init, "mirror_product", 4)
        full = ma.PromptState(init, "independent_cell", 4)
        small_raw = ma.payload(small, "mirror_product")
        full_raw = ma.payload(full, "independent_cell")
        self.assertLess(len(small_raw), len(full_raw))
        for raw in (small_raw, full_raw):
            head, tensor_bytes = raw.split(b"\n", 1)
            import json
            meta = json.loads(head)
            self.assertTrue(meta["tensor_manifest"])
            self.assertEqual(sum(__import__("math").prod(x["shape"]) * 4 for x in meta["tensor_manifest"]), len(tensor_bytes))


if __name__ == "__main__":
    unittest.main()
