"""MA-1162 L1 mechanism, dataset leakage and serialization tests."""
import io
import math
import torch
from torch.nn import functional as F
from source.l1 import QueryNet, TASKS, MODELS, rotate_pairs, make_world, make_batch, payload_bytes, param_counts

torch.set_num_threads(1)


def test_rotation_inverse_all_planes():
    x=torch.randn(3,4,32)
    a=torch.tensor(.45)
    assert torch.allclose(rotate_pairs(rotate_pairs(x,a),-a),x,atol=1e-6)


def test_exact_shared_qk_gauge():
    q,k=torch.randn(2,7,32),torch.randn(2,7,32)
    a=torch.tensor(.71)
    assert torch.allclose(q@k.transpose(-2,-1),rotate_pairs(q,a)@rotate_pairs(k,a).transpose(-2,-1),atol=5e-6)


def test_view_code_gradient_and_non_gauge():
    torch.manual_seed(3)
    model=QueryNet('mirror4')
    x=torch.randint(0,64,(4,32))
    F.cross_entropy(model(x),torch.randint(0,64,(4,))).backward()
    for n in ('q_code','k_code','v_code'):
        g=getattr(model,n).grad
        assert torch.isfinite(g).all() and g.abs().sum()>1e-8, (n,g)


def test_no_future_leakage_any_prefix_output():
    # Full causal logits must be unchanged before a perturbed suffix.
    torch.manual_seed(4)
    model=QueryNet('mirror4')
    p=make_world(618200)
    g=torch.Generator().manual_seed(20)
    x,y=make_batch('copy',3,g,p)
    assert x.shape==(3,32) and y.shape==(3,)
    x2=x.clone(); x2[:,20:]=torch.randint(0,64,x2[:,20:].shape)
    with torch.no_grad():
        a,b=model(x,return_all=True),model(x2,return_all=True)
    assert torch.equal(a[:,:20],b[:,:20])
    assert (a[:,20:]-b[:,20:]).abs().max()>0.01


def test_worlds_differ_and_reproduce():
    assert torch.equal(make_world(618200),make_world(618200))
    assert not torch.equal(make_world(618201),make_world(618202))


def test_content_retrieval_target_is_defined_without_future():
    g=torch.Generator().manual_seed(12)
    p=make_world(618200)
    x,y=make_batch('retrieve',256,g,p)
    inv=torch.argsort(p)
    raw=inv[x]
    assert torch.all(raw[:,16:-1]==63)
    qkey=raw[:,-1]-48
    keys=raw[:,:16]//3; vals=raw[:,:16]%3
    assert torch.all((keys==qkey[:,None]).sum(-1)==1)
    expected=(vals*(keys==qkey[:,None])).sum(-1)
    assert torch.equal(y,expected)


def test_order_not_permutation_invariant():
    # The noncommuting affine operators on Z3 yield different target states by order.
    def apply(ops):
        s=0
        for op in ops: s=(((op//3)+1)*s+(op%3))%3
        return s
    assert apply([1,3])!=apply([3,1])


def test_hidden_branch_is_input_independent():
    # Two calls with deterministic identical prefixes but independent target draws.
    g=torch.Generator().manual_seed(32)
    x,y=make_batch('hidden',4096,g,make_world(618200))
    inv=torch.argsort(make_world(618200))
    assert torch.all(inv[x[:,-1]]==62)
    assert (torch.bincount(y,minlength=3).float()/4096-1/3).abs().max()<.025


def test_roundtrip_and_bytes_all_models():
    x=torch.randint(0,64,(2,32))
    for kind in MODELS:
        model=QueryNet(kind).eval()
        raw=io.BytesIO()
        torch.save({'config':{'kind':kind,'d':32,'seq_len':32},'state':model.state_dict()},raw)
        assert len(raw.getvalue())==payload_bytes(model)
        restored=QueryNet(kind).eval()
        restored.load_state_dict(torch.load(io.BytesIO(raw.getvalue()),weights_only=True)['state'])
        with torch.no_grad(): assert torch.equal(model(x),restored(x))
        assert param_counts(model)[0]>0


def test_copy_target_is_causal_query():
    p=make_world(618200)
    g=torch.Generator().manual_seed(52)
    x,y=make_batch('copy',512,g,p)
    r=torch.argsort(p)[x]
    lag=r[:,-1]-48
    assert torch.equal(y,r.gather(1,(31-lag)[:,None]).squeeze(1))
