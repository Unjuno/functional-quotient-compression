import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from model import Intervention, RANK, givens, make_batch, teacher_params


class MA504Tests(unittest.TestCase):
    def test_givens_orthogonal_and_grad(self):
        a = torch.randn(3, 8, 6, requires_grad=True)
        r = givens(a)
        ident = torch.eye(RANK).expand_as(r)
        self.assertTrue(torch.allclose(r.transpose(-1, -2) @ r, ident, atol=1e-6))
        r.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_context_changes_target(self):
        x, c, y = make_batch(50400, "dev", 16)
        a, b, angles = teacher_params(50400)
        z = F.linear(x, a)
        flipped = torch.einsum("btij,btj->bti", givens(angles[(1-c).long()]), z)
        y_flipped = F.linear(flipped, b)
        self.assertGreater(float((y-y_flipped).abs().mean()), 1e-3)
        self.assertEqual(y.shape, (16, 8, 8))

    def test_variants_forward_and_grad(self):
        x, c, y = make_batch(50401, "train", 4)
        for mode in ("static", "independent", "mirror", "film", "full"):
            model = Intervention(mode, 11)
            pred = model(x, c)
            self.assertEqual(pred.shape, y.shape)
            loss = F.mse_loss(pred, y); loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mirror_and_film_generator_are_contextual(self):
        x = torch.randn(2, 8, 16)
        c = torch.zeros(2, 8)
        c[1] = 1
        for mode in ("mirror", "film"):
            model = Intervention(mode, 13)
            with torch.no_grad():
                model.b.normal_()
                model.generator[-1].weight.normal_()
            out = model(x, c)
            self.assertGreater(float((out[0]-out[1]).abs().mean()), 1e-5)


if __name__ == "__main__":
    unittest.main()
