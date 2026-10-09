import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from transcoder_ops import forward, nonoverlap_starts

class TranscoderOpsTest(unittest.TestCase):
    def test_nonoverlap_sampling_is_reproducible(self):
        a=nonoverlap_starts(10000,32,53301)
        self.assertEqual(a,nonoverlap_starts(10000,32,53301))
        self.assertEqual(len(a),32)
        self.assertTrue(all(y-x>=128 for x,y in zip(a,a[1:])))

    def test_matches_sparsecoder_formula_and_topk(self):
        torch.manual_seed(11)
        d, f, n, k = 5, 13, 7, 4
        x=torch.randn(n,d)
        w={'encoder.weight':torch.randn(f,d),'encoder.bias':torch.randn(f),
           'W_dec':torch.randn(f,d),'b_dec':torch.randn(d),'W_skip':torch.randn(d,d)}
        got, acts, ids=forward(x,w,k)
        pre=x@w['encoder.weight'].T+w['encoder.bias']
        vals, ix=pre.topk(k,dim=-1)
        ref=(vals.clamp_min(0).unsqueeze(-1)*w['W_dec'][ix]).sum(dim=1)+w['b_dec']+x@w['W_skip'].T
        self.assertTrue(torch.equal(ids,ix))
        self.assertTrue(torch.equal(acts,vals.clamp_min(0)))
        self.assertTrue(torch.allclose(got,ref,atol=0,rtol=0))

if __name__=='__main__': unittest.main()
