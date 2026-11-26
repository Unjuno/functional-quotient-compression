from types import SimpleNamespace
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from ma1059 import OutputLoRA


def test_inactive_hook_does_not_leak_into_next_view():
    layer=torch.nn.Linear(256,4096,bias=False)
    pipe=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(lm_head=layer)))
    a=OutputLoRA(pipe); b=OutputLoRA(pipe)
    with torch.no_grad():
        a.B.fill_(.001); b.B.fill_(.002)
    x=torch.ones(1,256); base=layer(x).detach()
    a.set('mirror',torch.tensor([.2])); b.set('inactive',None)
    only_a=layer(x).detach()
    b.set('mirror',torch.tensor([.3])); a.set('inactive',None)
    only_b=layer(x).detach()
    a.set('inactive',None); b.set('inactive',None)
    inactive=layer(x).detach()
    assert torch.equal(inactive,base)
    assert not torch.equal(only_a,base)
    assert not torch.equal(only_b,base)
    assert not torch.equal(only_a,only_b)
    a.close(); b.close()
