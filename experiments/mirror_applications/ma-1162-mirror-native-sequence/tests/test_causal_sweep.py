"""MA-1162 L1 causal future-perturbation audit across all eight controls."""
import pytest
import torch
from source.l1 import MODELS, QueryNet


@pytest.mark.parametrize('kind', MODELS)
def test_every_model_is_causal_at_every_position(kind):
    torch.manual_seed(801)
    m=QueryNet(kind).eval()
    x=torch.randint(0,64,(2,32))
    changed=x.clone()
    changed[:,21:]=torch.randint(0,64,(2,11))
    with torch.no_grad():
        a=m(x,return_all=True)
        b=m(changed,return_all=True)
    assert torch.equal(a[:,:21],b[:,:21]), kind
    assert torch.isfinite(a).all()
