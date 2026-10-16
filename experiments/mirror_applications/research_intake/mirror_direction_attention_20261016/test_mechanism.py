import torch
from torch.nn import functional as F
from pilot import B, D, K, R, U, base, factorized, full_joint, repeated, codes


def test_all_methods_output_shapes():
    x = torch.randn(B, 19, D)
    assert base(x).shape == (B,19,D)
    for fn in (factorized,full_joint,repeated):
        assert fn(x).shape == (B,19,K,D)


def test_factorized_and_joint_do_not_access_future_tokens():
    torch.manual_seed(901)
    x=torch.randn(B,19,D)
    changed=x.clone();changed[:,11:]+=9.0
    for fn in (factorized,full_joint):
        assert torch.allclose(fn(x)[:,:11],fn(changed)[:,:11], atol=2e-6,rtol=0)


def test_view0_receives_gradient_from_other_direction_codes():
    torch.manual_seed(902)
    t=20;x=torch.randn(B,t,D)
    c=codes.clone().requires_grad_(True)
    z=(base(x)@U)[:,:,None,:]*c[None,None,:,:]
    mix=F.scaled_dot_product_attention(z.reshape(B*t,1,K,R),z.reshape(B*t,1,K,R),z.reshape(B*t,1,K,R)).reshape(B,t,K,R)
    grad=torch.autograd.grad((mix[:,:,0,:]**2).mean(),c)[0]
    assert (grad.norm(dim=1)>1e-8).all().item()


def test_uniform_direction_codes_collapse_to_one_function():
    torch.manual_seed(903)
    t=20;x=torch.randn(B,t,D);same=torch.ones(K,R)
    bs=base(x)
    z=(bs@U)[:,:,None,:]*same[None,None,:,:]
    mix=F.scaled_dot_product_attention(z.reshape(B*t,1,K,R),z.reshape(B*t,1,K,R),z.reshape(B*t,1,K,R)).reshape(B,t,K,R)
    out=bs[:,:,None,:]+(mix@U.T)
    assert torch.allclose(out[:,:,0,:],out[:,:,1,:],atol=1e-6)
