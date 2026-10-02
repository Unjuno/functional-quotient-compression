# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Run: python -m unittest -v test_model. CPU numerical correctness tests."""
import io
import unittest
import torch
import torch.nn.functional as F
import model

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.assertTrue(hasattr(model, 'MirrorMLP'), 'MirrorMLP not implemented')

    def test_native_equals_explicit_expert_outputs_and_gradients(self):
        ff = model.MirrorMLP(8, 12, 4).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        y, p = ff(x)
        h = F.gelu(ff.up(x))
        states = 1 + ff.states
        ys = torch.stack([ff.down(h * g) for g in states], dim=-2)
        explicit = (ys * p.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(y, explicit, rtol=1e-10, atol=1e-10)
        g1 = torch.autograd.grad(y.square().sum(), (x, *ff.parameters()), retain_graph=True)
        g2 = torch.autograd.grad(explicit.square().sum(), (x, *ff.parameters()))
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)

    def test_all_components_receive_gradients(self):
        ff = model.MirrorMLP(8, 12, 4)
        ff(torch.randn(4, 3, 8))[0].square().mean().backward()
        for name, p in ff.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_parameter_count_and_no_full_expert_matrices(self):
        ff = model.MirrorMLP(8, 12, 4)
        expected = 2 * 8 * 12 + 12 + 8 + 4 * (8 + 1) + 4 * 12
        self.assertEqual(sum(p.numel() for p in ff.parameters()), expected)
        self.assertEqual(len([m for m in ff.modules() if isinstance(m, torch.nn.Linear)]), 3)
        self.assertTrue(all(p.ndim <= 2 for p in ff.parameters()))

    def test_permuting_state_labels_preserves_function(self):
        ff = model.MirrorMLP(8, 12, 4)
        x = torch.randn(2, 3, 8)
        y = ff(x)[0]
        perm = torch.tensor([3, 2, 0, 1])
        with torch.no_grad():
            ff.states.copy_(ff.states[perm])
            ff.router.weight.copy_(ff.router.weight[perm])
            ff.router.bias.copy_(ff.router.bias[perm])
        torch.testing.assert_close(y, ff(x)[0])

    def test_uniform_ablation_and_single_state(self):
        ff = model.MirrorMLP(8, 12, 4)
        x = torch.randn(2, 3, 8)
        y, p = ff(x, 'uniform')
        torch.testing.assert_close(p, torch.full_like(p, 0.25))
        expected = ff.down(F.gelu(ff.up(x)) * (1 + ff.states.mean(0)))
        torch.testing.assert_close(y, expected)
        one = model.MirrorMLP(8, 12, 1)
        torch.testing.assert_close(one(x)[1], torch.ones(2, 3, 1))

    def test_full_forward_matches_token_and_chunk_cache(self):
        lm = model.TinyLM(d=16, hidden=24, states=4, layers=2, heads=4).eval()
        ids = torch.randint(0, 21, (2, 13))
        with torch.no_grad():
            full = lm(ids)[0]
            for chunks in ([1]*13, [3,4,2,4]):
                past, pos, out = None, 0, []
                for chunk in chunks:
                    z, past, _ = lm(ids[:, pos:pos+chunk], past=past)
                    out.append(z); pos += chunk
                torch.testing.assert_close(full, torch.cat(out, 1), rtol=2e-5, atol=2e-5)
                for k, v in past:
                    self.assertEqual(k.shape, (2, 4, 13, 4))
                    self.assertEqual(v.shape, k.shape)

    def test_no_future_token_leakage(self):
        lm = model.TinyLM(d=16, hidden=24, layers=2, heads=4).eval()
        ids = torch.randint(0, 21, (2, 15))
        changed = ids.clone(); changed[:, 7:] = torch.randint(0, 21, (2, 8))
        with torch.no_grad():
            torch.testing.assert_close(lm(ids)[0][:, :7], lm(changed)[0][:, :7], rtol=0, atol=0)

    def test_cache_size_independent_of_states(self):
        sizes = []
        for states in (1, 4, 16):
            lm = model.TinyLM(d=16, hidden=24, states=states, layers=2, heads=4)
            _, past, _ = lm(torch.zeros(2, 5, dtype=torch.long))
            sizes.append(sum(k.numel()+v.numel() for k,v in past))
        self.assertEqual(sizes, [640, 640, 640])

    def test_checkpoint_round_trip(self):
        lm = model.TinyLM(d=16, hidden=24, layers=2, heads=4).eval()
        f = io.BytesIO(); torch.save(lm.state_dict(), f); f.seek(0)
        other = model.TinyLM(d=16, hidden=24, layers=2, heads=4).eval()
        other.load_state_dict(torch.load(f, weights_only=True))
        x = torch.randint(0,21,(2,5))
        torch.testing.assert_close(lm(x)[0], other(x)[0], rtol=0, atol=0)

    def test_controls_forward_and_bad_configuration(self):
        for kind in ('dense', 'direct_gate', 'full_moe', 'mirror'):
            lm = model.TinyLM(kind=kind, d=16, hidden=24, layers=2, heads=4)
            out = lm(torch.zeros(2, 5, dtype=torch.long))[0]
            self.assertEqual(out.shape, (2, 5, 21))
            self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):
            model.TinyLM(d=15, heads=4)
        with self.assertRaises(ValueError):
            model.TinyLM(kind='invalid')
        with self.assertRaises(ValueError):
            model.MirrorMLP(8, 12, 0)

    def test_fixed_routing_distribution(self):
        ff = model.MirrorMLP(8, 12, 4)
        x = torch.randn(2, 3, 8)
        v = torch.tensor([0.1, 0.2, 0.3, 0.4])
        y, p = ff(x, v)
        torch.testing.assert_close(p, v.expand_as(p))
        torch.testing.assert_close(y, ff.down(F.gelu(ff.up(x)) * (1 + v @ ff.states)))
        for bad in (torch.tensor([1., 0., 0., -1.]), torch.ones(4), torch.tensor([float('nan')]*4)):
            with self.assertRaises(ValueError):
                ff(x, bad)

    def test_identical_states_are_a_router_collapse(self):
        ff = model.MirrorMLP(8, 12, 4).double()
        with torch.no_grad():
            ff.states.copy_(ff.states[0:1].expand_as(ff.states))
        x = torch.randn(3, 4, 8, dtype=torch.float64)
        a = ff(x)[0]
        b = ff(x, 'uniform')[0]
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        a.square().sum().backward()
        self.assertLess(ff.router.weight.grad.abs().max().item(), 1e-12)

    def test_exact_parameter_matched_dense(self):
        mirror = model.TinyLM(kind='mirror', hidden=32)
        dense = model.TinyLM(kind='dense', hidden=36)
        self.assertEqual(sum(p.numel() for p in mirror.parameters()), 18973)
        self.assertEqual(sum(p.numel() for p in dense.parameters()), 18973)

    def test_algorithmic_data_mask_and_disjoint_sequences(self):
        from task import dataset, make_table
        tables = make_table()
        x, y = dataset(512, 12001, tables)
        self.assertEqual(x.shape, (512, 12))
        self.assertTrue((y[:, 0::3] == -100).all())
        self.assertTrue((y[:, 1::3] == -100).all())
        torch.testing.assert_close(y[:, 2::3], tables[x[:, 1::3]-32, x[:, 2::3]])
        other = dataset(512, 23002, tables)[0]
        a = {tuple(t.tolist()) for t in x}
        b = {tuple(t.tolist()) for t in other}
        self.assertFalse(a & b)

    def test_cache_invalid_shapes_and_context_limit(self):
        lm = model.TinyLM(d=16, hidden=24, layers=2, heads=4, max_length=8)
        _, past, _ = lm(torch.zeros(2, 5, dtype=torch.long))
        with self.assertRaises(ValueError):
            lm(torch.zeros(2, 4, dtype=torch.long), past=past)
        with self.assertRaises(ValueError):
            lm(torch.zeros(3, 1, dtype=torch.long), past=past)
        with self.assertRaises(ValueError):
            lm(torch.zeros(2, 1, dtype=torch.long), past=past[:1])

if __name__ == '__main__':
    unittest.main()
