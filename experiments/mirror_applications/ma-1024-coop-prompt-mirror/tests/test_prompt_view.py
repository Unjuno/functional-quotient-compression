import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
import coop_mirror as c

def test_givens_view_preserves_context_norm_and_changes_view():
    x=torch.arange(4*8,dtype=torch.float32).reshape(4,8)
    y=c.rotate_context(x,torch.tensor(.7))
    assert torch.allclose(x.norm(),y.norm(),atol=1e-6)
    assert not torch.allclose(x,y)

def test_prompt_payload_counts_domain_codes():
    x=torch.nn.Parameter(torch.ones(4,512))
    shared=len(c.prompt_payload('shared_coop',x))
    mirror=len(c.prompt_payload('mirror_domain_phase',x,torch.zeros(4)))
    independent=len(c.prompt_payload('independent_domain_coop',x[None].repeat(4,1,1)))
    assert mirror>shared and independent>mirror

def test_native_scalar_control_is_same_mirror_parameterization():
    x=torch.arange(4*512,dtype=torch.float32).reshape(4,512);p=torch.tensor(.4)
    assert torch.equal(c.rotate_context(x,p),c.rotate_context(x,p))
