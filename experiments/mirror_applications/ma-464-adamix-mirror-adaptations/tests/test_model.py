import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from model import AdaptedGPT, MODES, make_gpt_config, rotation_matrix, _nano


class AdaMixMirrorTests(unittest.TestCase):
    def base(self, seed=464):
        torch.manual_seed(seed)
        config = make_gpt_config(17, block_size=16)
        model = _nano.GPT(config)
        return config, {k: v.detach().clone() for k, v in model.state_dict().items()}

    def test_rank4_givens_is_orthogonal_and_receives_gradients(self):
        raw_angles = torch.randn(6, requires_grad=True)
        angles = raw_angles * 0.05
        matrix = rotation_matrix(angles)
        torch.testing.assert_close(matrix.T @ matrix, torch.eye(4), atol=2e-6, rtol=2e-6)
        x = torch.randn(8, 4)
        torch.testing.assert_close(x.norm(dim=-1), (x @ matrix.T).norm(dim=-1), atol=2e-6, rtol=2e-6)
        (x @ matrix.T).square().sum().backward()
        self.assertIsNotNone(raw_angles.grad)

    def test_all_conditions_forward_and_adapter_only_gradients(self):
        config, state = self.base()
        ids = torch.randint(0, 17, (2, 16)); targets = torch.randint(0, 17, (2, 16))
        counts = {}
        for condition in MODES:
            model = AdaptedGPT(state, config, condition, 17)
            model.set_view(1 if condition != "single" else 0, count=False)
            logits, loss = model(ids, targets)
            self.assertEqual(tuple(logits.shape), (2, 16, 17))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            trainable = model.adapter_parameters()
            self.assertTrue(trainable and all(p.requires_grad and p.grad is not None for p in trainable))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainable))
            self.assertTrue(all(not p.requires_grad for adapter in model.adapters for p in adapter.base.parameters()))
            counts[condition] = sum(p.numel() for p in trainable)
        self.assertGreater(counts["adamix"], counts["mirror"])
        self.assertLess(counts["mirror"], counts["adamix"])
        self.assertEqual(counts["mirror"] - 4 * (4 * 64 + 3 * 64 * 4), 4 * 12)

    def test_folded_delta_equals_selected_view_function(self):
        config, state = self.base(465)
        tokens = torch.randint(0, 17, (2, 16))
        for condition in MODES:
            model = AdaptedGPT(state, config, condition, 17)
            # Make the low-rank branch nonzero so the equality is substantive.
            with torch.no_grad():
                for adapter in model.adapters:
                    if condition == "adamix":
                        adapter.b.normal_(0, 0.01)
                    else:
                        adapter.b.normal_(0, 0.01)
                    if condition == "mirror":
                        adapter.codes.normal_(0, 0.05)
                    if condition == "film":
                        adapter.codes.add_(0.05)
            view = 0
            model.set_view(view, count=False); model.eval()
            with torch.no_grad():
                adapted, _ = model(tokens)
                folded = model.merged_model((view,))
                folded_out, _ = folded(tokens)
            torch.testing.assert_close(adapted, folded_out, atol=3e-5, rtol=3e-5)
            # The mean-view merge stores the exact arithmetic mean QKV delta.
            merged = model.merged_model((0, 1) if condition != "single" else (0,))
            for layer, adapter in enumerate(model.adapters):
                ids = (0, 1) if condition != "single" else (0,)
                expected = torch.stack([adapter.view_delta(i) for i in ids]).mean(0)
                original = state[f"transformer.h.{layer}.attn.c_attn.weight"]
                observed = merged.transformer.h[layer].attn.c_attn.weight - original
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-6)

    def test_bank_state_serializes_without_duplicate_base_module_keys(self):
        config, state = self.base(466)
        model = AdaptedGPT(state, config, "mirror", 17)
        state_keys = list(model.state_dict())
        self.assertFalse(any("adapters." in key for key in state_keys))
        self.assertEqual(len([k for k in state_keys if k.endswith(".base.weight")]), 4)
        bank = model.bank_adapter_state()
        self.assertEqual(len(bank), 4 * 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.pt"
            torch.save({"state": model.state_dict(), "bank": bank}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            clone = AdaptedGPT(state, config, "mirror", 17)
            clone.load_state_dict(payload["state"])
            ids = torch.randint(0, 17, (2, 16))
            model.set_view(1, count=False); clone.set_view(1, count=False)
            with torch.no_grad():
                expected, _ = model(ids)
                actual, _ = clone(ids)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            self.assertEqual(path.stat().st_size, len(path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
