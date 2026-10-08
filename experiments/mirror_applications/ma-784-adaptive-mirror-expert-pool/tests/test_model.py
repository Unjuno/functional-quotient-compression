import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from model import CONDITIONS, SHAPES, ModelConfig, SmallGPT, apply_givens, estimate_normrouter_c
import run_development as runner


def config():
    return ModelConfig(layers=4, width=64, heads=4, experts=4, expert_hidden=128,
                       block_size=16, dropout=0.0, gate_groups=8, pool_aux_weight=0.01,
                       normrouter_c_by_logical={n: estimate_normrouter_c(n, samples=2000) for n in (2, 4)})


class AdaptivePoolTests(unittest.TestCase):
    def test_givens_is_norm_preserving_and_has_view_gradient(self):
        torch.manual_seed(784)
        x = torch.randn(7, 64)
        raw = torch.randn(7, 8, requires_grad=True)
        y = apply_givens(x, raw)
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1), atol=2e-5, rtol=2e-5)
        y[:, :16].sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_all_frozen_conditions_forward_backward_and_pool_sizes(self):
        torch.manual_seed(785)
        ids = torch.randint(0, 31, (2, 16)); targets = torch.randint(0, 31, (2, 16))
        counts = {}
        for condition in CONDITIONS:
            model = SmallGPT(config(), condition, 31)
            logits, loss, aux, loads, entropy = model(ids, targets)
            self.assertEqual(tuple(logits.shape), (2, 16, 31))
            self.assertTrue(torch.isfinite(loss + aux))
            self.assertTrue(torch.isfinite(loads).all())
            self.assertTrue(torch.isfinite(entropy))
            (loss + config().pool_aux_weight * aux).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            counts[condition] = sum(p.numel() for p in model.parameters())
            physical, logical, mode = SHAPES[condition]
            if mode in ("shared", "mirror", "film"):
                self.assertEqual(len(model.shared_experts), physical)
                self.assertEqual(model.layers[0].router.out_features, logical)
        self.assertLess(counts["mirror2x"], counts["unipool4"])
        self.assertEqual(counts["mirror2x"] - counts["hard_alias2x"], 4 * 4 * 8)
        self.assertGreater(counts["untied_moe"], counts["unipool4"])

    def test_distinct_codes_change_the_selected_expert_function(self):
        torch.manual_seed(786)
        cfg = config()
        model = SmallGPT(cfg, "mirror2x", 31)
        model.eval()
        h = torch.randn(12, 64)
        angles = torch.zeros(8)
        rotated = apply_givens(h, angles)
        torch.testing.assert_close(h, rotated)
        changed = apply_givens(h, torch.full((8,), 0.2))
        self.assertGreater(float((changed - h).abs().max()), 1e-3)

    def test_one_update_serializes_reloadable_inference_payload_without_data_download(self):
        torch.manual_seed(787)
        cfg = runner.make_config()
        model = SmallGPT(cfg, "mirror2x", 9)
        train = torch.randint(0, 9, (512,)); dev = torch.randint(0, 9, (512,))
        vocab = [str(i) for i in range(9)]
        manifest = {"sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            artifact_dir.mkdir()
            with patch.multiple(runner, UPDATES=1, EVAL_EVERY=1, EVAL_BATCHES=1,
                                ARTIFACTS=artifact_dir):
                result = runner.train_one(788, "mirror2x", model, train, dev, vocab, manifest)
            payload = torch.load(artifact_dir / "seed788_mirror2x_inference.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["condition"], "mirror2x")
            self.assertEqual(len(payload["vocabulary"]), len(vocab))
            self.assertEqual(result["inference_payload"]["bytes"], (artifact_dir / "seed788_mirror2x_inference.pt").stat().st_size)
            self.assertTrue(torch.isfinite(torch.tensor(result["nll"])))


if __name__ == "__main__":
    unittest.main()
