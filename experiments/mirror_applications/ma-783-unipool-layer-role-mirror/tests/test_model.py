import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from model import CONDITIONS, ModelConfig, SmallGPT, apply_givens, estimate_normrouter_c
from run_development import measure_role_diversity


def config():
    return ModelConfig(layers=4, width=64, heads=4, experts=4, expert_hidden=128,
                       block_size=16, dropout=0.0, gate_groups=8, depth_code_dim=8,
                       aux_weight=0.01, pool_aux_weight=0.01,
                       normrouter_c=estimate_normrouter_c(4, 1, samples=2000))


class UniPoolMirrorTests(unittest.TestCase):
    def test_givens_preserves_norm_and_has_finite_angle_gradient(self):
        torch.manual_seed(783)
        x = torch.randn(3, 5, 64)
        raw_angle = torch.randn(8, requires_grad=True)
        angle = raw_angle * 0.1
        y = apply_givens(x, angle)
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1), atol=2e-5, rtol=2e-5)
        y.square().sum().backward()
        self.assertIsNotNone(raw_angle.grad)
        self.assertTrue(torch.isfinite(y).all())

    def test_each_condition_forward_backward_and_shared_pool_structure(self):
        torch.manual_seed(784)
        cfg = config()
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        counts = {}
        for condition in CONDITIONS:
            model = SmallGPT(cfg, condition, 32)
            logits, lm_loss, aux, load, entropy = model(ids, targets)
            self.assertEqual(tuple(logits.shape), (2, 16, 32))
            self.assertTrue(torch.isfinite(lm_loss + aux))
            self.assertTrue(torch.isfinite(load).all())
            self.assertTrue(torch.isfinite(entropy))
            (lm_loss + cfg.aux_weight * aux).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            counts[condition] = sum(p.numel() for p in model.parameters())
            if condition in ("unipool", "mirror_givens", "film_gate", "depth_embedding"):
                self.assertEqual(len(model.shared_experts), 4)
                self.assertTrue(all(layer.experts is None for layer in model.layers))
            if condition == "untied_moe":
                self.assertTrue(all(len(layer.experts) == 4 for layer in model.layers))
        self.assertGreater(counts["untied_moe"], counts["unipool"])
        self.assertEqual(counts["unipool"] + 4 * 8, counts["mirror_givens"])

    def test_layer_role_diagnostic_uses_same_heldout_activations(self):
        torch.manual_seed(785)
        cfg=config();cfg.block_size=128
        model = SmallGPT(cfg, "mirror_givens", 32)
        tokens = torch.randint(0, 32, (256,))
        positions = torch.arange(16)
        distance = measure_role_diversity(model, tokens, positions)
        self.assertTrue(torch.isfinite(torch.tensor(distance)))
        self.assertGreaterEqual(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
