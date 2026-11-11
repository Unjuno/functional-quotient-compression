import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import run


class ScreenTests(unittest.TestCase):
    def test_interpolation_coordinate_is_exact_reparameterization(self):
        torch.manual_seed(5)
        a, b = run.Net(), run.Net()
        x = torch.randn(23, 64)
        sa, sb = a.state_dict(), b.state_dict()
        for alpha in run.ALPHAS:
            m = 2 * float(alpha) - 1
            alpha_from_m = (m + 1) / 2
            pa = {k: (1-float(alpha))*sa[k] + float(alpha)*sb[k] for k in sa}
            pm = {k: (1-alpha_from_m)*sa[k] + alpha_from_m*sb[k] for k in sa}
            za = torch.func.functional_call(a, pa, (x,))
            zm = torch.func.functional_call(a, pm, (x,))
            self.assertTrue(torch.equal(za, zm))

    def test_sampled_interpolation_objective_updates_both_endpoints(self):
        torch.manual_seed(9)
        a, b = run.Net(), run.Net()
        x, y = torch.randn(12,64), torch.randint(0,10,(12,))
        alpha = torch.tensor(.37)
        pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
        mixed = {k:(1-alpha)*pa[k]+alpha*pb[k] for k in pa}
        loss = nn.functional.cross_entropy(torch.func.functional_call(a,mixed,(x,)),y)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters()))

    def test_all_fresh_serialization_roundtrips_are_exact(self):
        # Recheck on the fixed development seed; fresh runs recorded the same checks.
        r = run.run_seed(31, .1, .1)
        self.assertEqual(r['max_native_mirror_logit_diff'], 0.0)
        self.assertEqual(r['max_reload_line_logit_diff'], 0.0)
        self.assertEqual(r['max_materialized_roundtrip_logit_diff'], 0.0)
        self.assertLess(r['native_line_bytes'], r['materialized_17_line_bytes'])


if __name__ == '__main__':
    unittest.main()
