import sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from model import METHODS,SpecSystem,rotate8,speculative_distribution  # noqa:E402
from run import make_world,save_payload,load_payload,score  # noqa:E402

def test_residual_corrected_speculation_is_exact():
    torch.manual_seed(4);p=F.softmax(torch.randn(32,16),dim=-1);q=F.softmax(torch.randn(32,16),dim=-1)
    out,a=speculative_distribution(q,p)
    torch.testing.assert_close(out,p,atol=2e-7,rtol=2e-7)
    assert bool(((a>=0)&(a<=1)).all())

def test_mirror_givens_preserves_hidden_norm():
    h=torch.randn(128,8);a=torch.randn(4)
    torch.testing.assert_close(rotate8(h,a).norm(dim=-1),h.norm(dim=-1),atol=2e-6,rtol=2e-6)

def test_nested_widths_are_supernet_prefixes():
    w=make_world(39901);s=SpecSystem('nested8',w['w1'],w['b1'],w['w2'],w['b2'])
    x=torch.randn(12,8)
    h8=torch.tanh(x@w['w1'][:,:8]+w['b1'][:8]);expected=h8@w['w2'][:8]+w['b2']
    torch.testing.assert_close(s.draft_logits(x),expected)

def test_all_payload_methods_roundtrip_scores(tmp_path):
    w=make_world(39902)
    for method in METHODS:
        s=SpecSystem(method,w['w1'],w['b1'],w['w2'],w['b2'])
        p=tmp_path/f'{method}.npz';save_payload(p,s);loaded=load_payload(p)
        assert loaded.method==method and score(loaded,w)==score(s,w)
